import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

# --- CONFIGURAZIONE PERCORSI ---
VAL_CSV = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_validation_results.csv"
TEST_CSV = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_detection_results.csv"

CTX_COLS = ['Ctx_Pos0', 'Ctx_Pos1', 'Ctx_Pos2', 'Ctx_Pos3', 'Ctx_Pos4']

# Tolleranze di Falsi Allarmi (FPR) da testare sul traffico sano
PERCENTILES_TO_TEST = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]

def main():
    print("Lettura dei dataset pre-calcolati...")
    try:
        df_val = pd.read_csv(VAL_CSV)
        df_test = pd.read_csv(TEST_CSV)
    except FileNotFoundError as e:
        print(f"Errore caricamento file: {e}")
        return

    # Pulizia Base: Riempiamo i NaN con la media della riga
    df_val[CTX_COLS] = df_val[CTX_COLS].apply(lambda row: row.fillna(row.mean()), axis=1)
    df_test[CTX_COLS] = df_test[CTX_COLS].apply(lambda row: row.fillna(row.mean()), axis=1)

    # 1. GARANZIA UNSUPERVISED: Usiamo solo i pacchetti SANI dal Validation
    if 'True_Label' in df_val.columns and df_val['True_Label'].sum() > 0:
        val_sani = df_val[df_val['True_Label'] == 0][CTX_COLS]
    else:
        val_sani = df_val[CTX_COLS]

    # --- CALCOLO DELLE STATISTICHE SUL VALIDATION (Traffico Sano) ---
    val_mean = val_sani.mean(axis=1).values
    val_min = val_sani.min(axis=1).values
    val_median = val_sani.median(axis=1).values
    val_std = val_sani.std(axis=1).fillna(0).values
    # Usiamo il valore assoluto della differenza tra inizio e fine finestra
    val_diff = (val_sani['Ctx_Pos0'] - val_sani['Ctx_Pos4']).abs().values

    # --- CALCOLO DELLE STATISTICHE SUL TEST ---
    y_test = df_test['True_Label'].values
    test_mean = df_test[CTX_COLS].mean(axis=1).values
    test_min = df_test[CTX_COLS].min(axis=1).values
    test_median = df_test[CTX_COLS].median(axis=1).values
    test_std = df_test[CTX_COLS].std(axis=1).fillna(0).values
    test_diff = (df_test['Ctx_Pos0'] - df_test['Ctx_Pos4']).abs().values

    print("\nRicerca della soglia ottimale per SINGOLA STRATEGIA ESTRATTIVA in corso...\n")
    
    # Struttura per salvare i vincitori
    strategies = {
        'Media': {'f1': -1, 'percentile': 0, 'pred': None, 'threshold': 0, 'scores_for_auc': -test_mean},
        'Minimo': {'f1': -1, 'percentile': 0, 'pred': None, 'threshold': 0, 'scores_for_auc': -test_min},
        'Mediana': {'f1': -1, 'percentile': 0, 'pred': None, 'threshold': 0, 'scores_for_auc': -test_median},
        'Dev_Standard': {'f1': -1, 'percentile': 0, 'pred': None, 'threshold': 0, 'scores_for_auc': test_std},
        'Diff_Pos0_Pos4': {'f1': -1, 'percentile': 0, 'pred': None, 'threshold': 0, 'scores_for_auc': test_diff}
    }

    # --- FASE 1: GRID SEARCH SULLE TOLLERANZE ---
    for fpr in PERCENTILES_TO_TEST:
        # Code inferiori (Valori più bassi = Anomalia)
        th_mean = np.percentile(val_mean, fpr)
        th_min = np.percentile(val_min, fpr)
        th_median = np.percentile(val_median, fpr)
        
        # Code superiori (Alta Varianza/Differenza = Anomalia)
        th_std = np.percentile(val_std, 100.0 - fpr)
        th_diff = np.percentile(val_diff, 100.0 - fpr)

        # Predizioni Test
        preds = {
            'Media': (test_mean < th_mean).astype(int),
            'Minimo': (test_min < th_min).astype(int),
            'Mediana': (test_median < th_median).astype(int),
            'Dev_Standard': (test_std > th_std).astype(int),
            'Diff_Pos0_Pos4': (test_diff > th_diff).astype(int)
        }

        # Soglie associate per logging
        thresholds = {
            'Media': th_mean, 'Minimo': th_min, 'Mediana': th_median, 
            'Dev_Standard': th_std, 'Diff_Pos0_Pos4': th_diff
        }

        # Valutazione F1-Score
        for name in strategies.keys():
            current_f1 = f1_score(y_test, preds[name], zero_division=0)
            
            if current_f1 > strategies[name]['f1']:
                strategies[name].update({
                    'f1': current_f1,
                    'percentile': fpr,
                    'pred': preds[name],
                    'threshold': thresholds[name]
                })

    # --- FASE 2: STAMPA DEI REPORT ---
    print("="*75)
    print(" REPORT COMPLETO (OGNI STRATEGIA CON LA SUA SOGLIA MIGLIORE)")
    print("="*75)
    
    overall_best_f1 = -1
    overall_best_name = ""
    
    for name, data in strategies.items():
        y_pred = data['pred']
        f1 = data['f1']
        perc = data['percentile']
        thresh = data['threshold']
        scores_auc = data['scores_for_auc']
        
        # Traccia il vincitore assoluto
        if f1 > overall_best_f1:
            overall_best_f1 = f1
            overall_best_name = name
            
        try:
            auc = roc_auc_score(y_test, scores_auc)
        except ValueError:
            auc = 0.0

        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n{'-'*45}")
        print(f" Strategia: {name.upper()}")
        print(f" (Soglia Ottimale: {thresh:.4f} tolleranza FPR: {perc}%)")
        print(f"{'-'*45}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4, target_names=["Benign", "Attack"], zero_division=0))
        print("Confusion Matrix:")
        print(f"[TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}]")
        print(f"[FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}]")
        print(f"ROC AUC: {auc:.4f}")

    print("\n" + "="*75)
    print(f" 🏆 VINCITORE ASSOLUTO: {overall_best_name.upper()} (F1-Score: {overall_best_f1:.4f})")
    print("="*75)

if __name__ == "__main__":
    main()