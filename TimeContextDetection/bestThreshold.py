import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURAZIONE ---
# Inserisci qui il percorso del file CSV appena generato da detection.py
INPUT_FILE = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/detection_topK_final.csv"

def print_custom_report(y_true, y_pred, title):
    """Stampa il report formattato con Precision, Recall, F1 e Confusion Matrix"""
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"{'':<14} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}")
    r0 = report_dict['0']
    r1 = report_dict['1']
    print(f"{'Benign':<14} {r0['precision']:>9.4f} {r0['recall']:>9.4f} {r0['f1-score']:>9.4f} {r0['support']:>9}")
    print(f"{'Attack':<14} {r1['precision']:>9.4f} {r1['recall']:>9.4f} {r1['f1-score']:>9.4f} {r1['support']:>9}")
    print(f"\nConfusion Matrix:\n[TP: {tp:<5} | FN: {fn:<5}]\n[FP: {fp:<5} | TN: {tn:<5}]")
    return r1['f1-score']

def run_voting_strategy(df, strategy_name):
    """Cerca la soglia che massimizza l'F1-Score per la strategia specificata"""
    y_true = df['True_Label'].values
    
    # Seleziona solo le colonne con le probabilità (ignoriamo Voting_Pred che era quella vecchia)
    score_cols = [f'LogProb_Pos{i}' for i in range(5)]
    scores = df[score_cols]
    
    vals = scores.values.flatten()
    vals = vals[~np.isnan(vals)]
    
    # Generiamo 100 possibili soglie basate sulla distribuzione dei dati Top-K
    thresholds = np.linspace(np.percentile(vals, 1), np.percentile(vals, 99), 100)
    
    best_f1, best_th, best_preds = -1, None, None
    
    for th in thresholds:
        # CONDIZIONE DI ANOMALIA: LogProb < Soglia
        is_anomalous = scores < th 
        
        # --- LOGICHE DI VOTING ---
        if strategy_name == 'majority':
            vote = (is_anomalous.sum(axis=1) > (scores.notna().sum(axis=1) / 2)).astype(int)
        elif strategy_name == 'at_least_1':
            vote = is_anomalous.any(axis=1).astype(int)
        elif strategy_name == 'at_least_2':
            vote = (is_anomalous.sum(axis=1) >= 2).astype(int)
        elif strategy_name == 'strict_all':
            all_match = (is_anomalous.sum(axis=1) == scores.notna().sum(axis=1))
            has_data = scores.notna().sum(axis=1) > 0
            vote = (all_match & has_data).astype(int)
        
        # Calcolo F1
        f1 = classification_report(y_true, vote, output_dict=True)['1']['f1-score']
        
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_preds = vote
            
    return best_f1, best_th, best_preds

def main():
    print(f"Caricamento dati (Metrica Top-3) da {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("❌ File non trovato. Verifica il percorso.")
        return

    print("\n🔍 RICERCA BEST THRESHOLD PER STRATEGIA (Senza rifare inferenza!)")
    
    strategies = ['majority', 'at_least_1', 'at_least_2', 'strict_all']
    results = {}

    for strat in strategies:
        f1, th, preds = run_voting_strategy(df, strat)
        results[strat] = {'f1': f1, 'th': th, 'preds': preds}
        print(f"Strategia {strat.upper():<12} -> Miglior F1: {f1:.4f} (Soglia Ottimale: {th:.4f})")

    best_strat = max(results, key=lambda k: results[k]['f1'])
    
    print("\n" + "#"*60)
    print(" RISULTATI DETTAGLIATI ")
    print("#"*60)

    for strat in strategies:
        print_custom_report(df['True_Label'], results[strat]['preds'], f"STRATEGIA: {strat.upper()} (Soglia: {results[strat]['th']:.4f})")

    print(f"\n🏆 VINCITORE ASSOLUTO: {best_strat.upper()} con F1-Score: {results[best_strat]['f1']:.4f}")

if __name__ == "__main__":
    main()