import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

# --- CONFIGURAZIONE PERCORSI ---
VAL_CSV = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_validation_results.csv"
TEST_CSV = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_detection_results.csv"

CTX_COLS = ['Ctx_Pos0', 'Ctx_Pos1', 'Ctx_Pos2', 'Ctx_Pos3', 'Ctx_Pos4']

# Range di percentili più ampio e granulare (da 0.1% a 20%)
PERCENTILES_TO_TEST = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]

def main():
    print("Lettura dei dataset pre-calcolati...")
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)
    
    val_scores = df_val[CTX_COLS].values.flatten()
    val_scores = val_scores[~np.isnan(val_scores)]
    
    y_test = df_test['True_Label'].values
    test_scores = df_test[CTX_COLS]
    valid_counts = test_scores.notna().sum(axis=1)
    
    print("\nRicerca della soglia ottimale per SINGOLA STRATEGIA in corso...\n")
    
    # Dizionario per salvare i risultati migliori per ogni strategia
    best_results_per_strategy = {
        'at_least_1': {'f1': -1, 'percentile': 0, 'pred': None, 'threshold': 0},
        'at_least_2': {'f1': -1, 'percentile': 0, 'pred': None, 'threshold': 0},
        'at_least_4': {'f1': -1, 'percentile': 0, 'pred': None, 'threshold': 0},
        'majority': {'f1': -1, 'percentile': 0, 'pred': None, 'threshold': 0},
        'strict_all': {'f1': -1, 'percentile': 0, 'pred': None, 'threshold': 0}
    }

    # --- FASE 1: RICERCA DEL PERCENTILE OTTIMALE PER OGNI STRATEGIA ---
    for percentile in PERCENTILES_TO_TEST:
        threshold = np.percentile(val_scores, percentile)
        is_anomalous = test_scores < threshold
        anomaly_votes = is_anomalous.sum(axis=1)
        
        strategies_temp = {
            'at_least_1': (anomaly_votes >= 1).astype(int),
            'at_least_2': (anomaly_votes >= 2).astype(int),
            'at_least_4': (anomaly_votes >= 4).astype(int),
            'majority': (anomaly_votes > (valid_counts / 2)).astype(int),
            'strict_all': ((anomaly_votes == valid_counts) & (valid_counts > 0)).astype(int)
        }
        
        for name, y_pred in strategies_temp.items():
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Se questa combinazione (percentile + strategia) batte il record precedente di quella strategia, salvala
            if f1 > best_results_per_strategy[name]['f1']:
                best_results_per_strategy[name]['f1'] = f1
                best_results_per_strategy[name]['percentile'] = percentile
                best_results_per_strategy[name]['threshold'] = threshold
                best_results_per_strategy[name]['pred'] = y_pred

    # --- FASE 2: STAMPA DEI REPORT ---
    print("="*70)
    print(" REPORT COMPLETO (OGNI STRATEGIA CON LA SUA SOGLIA MIGLIORE)")
    print("="*70)
    
    overall_best_f1 = -1
    overall_best_name = ""
    
    for name, data in best_results_per_strategy.items():
        y_pred = data['pred']
        f1 = data['f1']
        perc = data['percentile']
        thresh = data['threshold']
        
        # Aggiorna il vincitore assoluto
        if f1 > overall_best_f1:
            overall_best_f1 = f1
            overall_best_name = name
            
        auc = roc_auc_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n{'-'*40}")
        print(f" Strategia: {name.upper()}")
        print(f" (Soglia Ottimale: {thresh:.4f} al {perc}% percentile)")
        print(f"{'-'*40}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        print("Confusion Matrix:")
        print(f"TP: {cm[1][1]:<6} | FN: {cm[1][0]:<6}")
        print(f"FP: {cm[0][1]:<6} | TN: {cm[0][0]:<6}")
        print(f"ROC AUC: {auc:.4f}")

    print("\n" + "="*70)
    print(f"🏆 VINCITORE ASSOLUTO: {overall_best_name.upper()} (F1-Score: {overall_best_f1:.4f})")
    print("="*70)

if __name__ == "__main__":
    main()