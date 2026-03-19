import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

# --- CONFIGURAZIONE PERCORSI ---
TEST_CSV = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_detection_results.csv"

# Colonne degli score contestuali
CTX_COLS = ['Ctx_Pos0', 'Ctx_Pos1', 'Ctx_Pos2', 'Ctx_Pos3', 'Ctx_Pos4']

def main():
    print(f"Lettura del dataset da: {TEST_CSV}...")
    df_test = pd.read_csv(TEST_CSV)
    
    y_true = df_test['True_Label'].values
    test_scores = df_test[CTX_COLS]
    valid_counts = test_scores.notna().sum(axis=1)
    
    # Range di ricerca per le log-probabilità dal Test Set (escludendo gli outlier estremi con i percentili 1 e 99)
    vals = test_scores.values.flatten()
    vals = vals[~np.isnan(vals)]
    thresholds = np.linspace(np.percentile(vals, 1), np.percentile(vals, 99), 100)
    
    print("\n🔍 Ricerca della soglia ottimale (Oracle) sul Test Set per ogni strategia...")
    
    # Inizializziamo il dizionario per salvare i risultati
    strategies_list = ['at_least_1', 'at_least_2', 'majority', 'at_least_4', 'strict_all']
    best_results = {s: {'f1': -1, 'threshold': 0, 'pred': None} for s in strategies_list}

    # Iteriamo su tutte le soglie per trovare la migliore per ogni strategia
    for th in thresholds:
        is_anomalous = test_scores < th
        anomaly_votes = is_anomalous.sum(axis=1)
        
        # Calcolo dei voti per ogni strategia
        preds = {
            'at_least_1': (anomaly_votes >= 1).astype(int),
            'at_least_2': (anomaly_votes >= 2).astype(int),
            'majority': (anomaly_votes > (valid_counts / 2)).astype(int),
            'at_least_4': (anomaly_votes >= 4).astype(int),
            'strict_all': ((anomaly_votes == valid_counts) & (valid_counts > 0)).astype(int)
        }
        
        # Valutazione F1-Score per ogni strategia con la soglia corrente
        for name, pred in preds.items():
            f1 = f1_score(y_true, pred, zero_division=0)
            if f1 > best_results[name]['f1']:
                best_results[name]['f1'] = f1
                best_results[name]['threshold'] = th
                best_results[name]['pred'] = pred

    # --- STAMPA DEI REPORT ---
    print("\n" + "="*70)
    print(" REPORT COMPLETO (UPPER BOUND SUL TEST SET)")
    print("="*70)
    
    overall_best_f1 = -1
    overall_best_name = ""
    
    for name in strategies_list:
        data = best_results[name]
        y_pred = data['pred']
        f1 = data['f1']
        thresh = data['threshold']
        
        # Aggiorna il vincitore assoluto
        if f1 > overall_best_f1:
            overall_best_f1 = f1
            overall_best_name = name
            
        auc = roc_auc_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\n{'-'*40}")
        print(f" Strategia: {name.upper()}")
        print(f" (Soglia Ottimale: {thresh:.4f})")
        print(f"{'-'*40}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))
        print("Confusion Matrix:")
        print(f"TP: {cm[1][1]:<6} | FN: {cm[1][0]:<6}")
        print(f"FP: {cm[0][1]:<6} | TN: {cm[0][0]:<6}")
        print(f"ROC AUC: {auc:.4f}")

    print("\n" + "="*70)
    print(f"🏆 VINCITORE ASSOLUTO: {overall_best_name.upper()} (Max F1-Score Teorico: {overall_best_f1:.4f})")
    print("="*70)

if __name__ == "__main__":
    main()