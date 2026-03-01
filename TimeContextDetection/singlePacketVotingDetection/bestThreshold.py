import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURAZIONE ---
# Assicurati che il file sia nella stessa cartella dello script o specifica il percorso completo
INPUT_FILE = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/detection_final_with_preds.csv"

def print_custom_report(y_true, y_pred, title):
    """Stampa il report formattato con Precision, Recall, F1 e Confusion Matrix"""
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"{'':<14} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}")
    
    if '0' in report_dict:
        r0 = report_dict['0']
        print(f"{'Benign':<14} {r0['precision']:>9.4f} {r0['recall']:>9.4f} {r0['f1-score']:>9.4f} {r0['support']:>9}")
    
    if '1' in report_dict:
        r1 = report_dict['1']
        print(f"{'Attack':<14} {r1['precision']:>9.4f} {r1['recall']:>9.4f} {r1['f1-score']:>9.4f} {r1['support']:>9}")
    
    print(f"\nConfusion Matrix:\n[TP: {tp:<5} | FN: {fn:<5}]\n[FP: {fp:<5} | TN: {tn:<5}]")
    
    return report_dict['1']['f1-score'] if '1' in report_dict else 0.0

def run_voting_strategy(df, strategy_name):
    """Cerca la soglia che massimizza l'F1-Score per la strategia specificata"""
    y_true = df['True_Label'].values
    
    # Seleziona solo le colonne con le probabilità LogProb_Pos0 ... LogProb_Pos4
    score_cols = [f'LogProb_Pos{i}' for i in range(5)]
    scores = df[score_cols]
    
    # Appiattisce i valori per calcolare i percentili, ignorando i NaN
    vals = scores.values.flatten()
    vals = vals[~np.isnan(vals)]
    
    # Generiamo 100 possibili soglie basate sulla distribuzione dei dati
    thresholds = np.linspace(np.percentile(vals, 1), np.percentile(vals, 99), 100)
    
    best_f1, best_th, best_preds = -1, None, None
    
    for th in thresholds:
        # CONDIZIONE DI ANOMALIA: LogProb < Soglia
        # Nota: con i NaN, il confronto restituisce False (che è corretto, non votano anomalia)
        is_anomalous = scores < th 
        
        # --- LOGICHE DI VOTING ---
        if strategy_name == 'majority':
            # Vota anomalia se > 50% dei pacchetti validi sono anomali
            valid_votes = scores.notna().sum(axis=1)
            # Gestione divisione per zero implicita (0 > 0 è False -> Benign)
            vote = (is_anomalous.sum(axis=1) > (valid_votes / 2)).astype(int)
            
        elif strategy_name == 'at_least_1':
            # Basta 1 anomalia
            vote = is_anomalous.any(axis=1).astype(int)
            
        elif strategy_name == 'at_least_2':
            # Almeno 2 anomalie
            vote = (is_anomalous.sum(axis=1) >= 2).astype(int)
            
        elif strategy_name == 'strict_all':
            # Tutti i pacchetti validi devono essere anomali (e deve esserci almeno un pacchetto)
            valid_count = scores.notna().sum(axis=1)
            anomalous_count = is_anomalous.sum(axis=1)
            all_match = (anomalous_count == valid_count)
            has_data = valid_count > 0
            vote = (all_match & has_data).astype(int)
        
        # Calcolo F1 su una copia per evitare warning
        try:
            report = classification_report(y_true, vote, output_dict=True, zero_division=0)
            f1 = report['1']['f1-score'] if '1' in report else 0.0
        except Exception:
            f1 = 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_preds = vote
            
    return best_f1, best_th, best_preds

def main():
    print(f"Caricamento dati da {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
        # Verifica veloce delle colonne
        required_cols = ['True_Label'] + [f'LogProb_Pos{i}' for i in range(5)]
        if not all(col in df.columns for col in required_cols):
            print(f"❌ Errore: Il CSV deve contenere le colonne: {required_cols}")
            return
    except FileNotFoundError:
        print(f"❌ File '{INPUT_FILE}' non trovato. Assicurati che sia nella stessa cartella.")
        return

    print("\n🔍 RICERCA BEST THRESHOLD PER STRATEGIA")
    
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
        print_custom_report(df['True_Label'], results[strat]['preds'], 
                          f"STRATEGIA: {strat.upper()} (Soglia: {results[strat]['th']:.4f})")

    print(f"\n🏆 VINCITORE ASSOLUTO: {best_strat.upper()} con F1-Score: {results[best_strat]['f1']:.4f}")

if __name__ == "__main__":
    main()