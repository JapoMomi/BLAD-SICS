import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# --- CONFIGURAZIONE ---
INPUT_FILE = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/singlePacketVotingDetection/detection_final_with_preds.csv"

def print_custom_report(y_true, y_pred, title):
    """Stampa il report formattato stile sklearn standard + Confusion Matrix + ROC AUC"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # Stampa il report standard di sklearn che include tutto (precision, recall, f1, support, averages)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"], digits=4))
    
    # Calcolo Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("Confusion Matrix:")
    print(f"TP: {tp:<5} | FN: {fn:<5}")
    print(f"FP: {fp:<5} | TN: {tn:<5}")
    
    # Calcolo ROC AUC (sulle predizioni binarie)
    try:
        auc = roc_auc_score(y_true, y_pred)
        print(f"ROC AUC: {auc:.4f}")
    except ValueError:
        print("ROC AUC: N/A (Solo una classe presente?)")

def get_voting_prediction(scores, threshold, strategy_name):
    """
    Calcola le predizioni (0 o 1) basate sulla strategia e sulla soglia data.
    Restituisce un array numpy di predizioni.
    """
    # CONDIZIONE DI ANOMALIA: LogProb < Soglia
    is_anomalous = scores < threshold
    
    # Conta quanti pacchetti validi (non NaN) ci sono per ogni riga
    valid_votes = scores.notna().sum(axis=1)
    
    # Conta quante anomalie ci sono per ogni riga
    anomalous_count = is_anomalous.sum(axis=1)

    if strategy_name == 'majority':
        # Vota anomalia se > 50% dei pacchetti validi sono anomali
        # Gestione divisione per zero implicita (0 > 0 è False -> Benign)
        return (anomalous_count > (valid_votes / 2)).astype(int)
        
    elif strategy_name == 'at_least_1':
        # Basta 1 anomalia
        return is_anomalous.any(axis=1).astype(int)
        
    elif strategy_name == 'at_least_2':
        # Almeno 2 anomalie
        return (anomalous_count >= 2).astype(int)
        
    elif strategy_name == 'strict_all':
        # Tutti i pacchetti validi devono essere anomali (e deve esserci almeno un pacchetto)
        all_match = (anomalous_count == valid_votes)
        has_data = valid_votes > 0
        return (all_match & has_data).astype(int)
    
    return np.zeros(len(scores), dtype=int)

def run_voting_strategy_optimization(df, strategy_name):
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
    
    best_f1 = -1
    best_th = 0.0
    best_preds = None
    
    for th in thresholds:
        preds = get_voting_prediction(scores, th, strategy_name)
        
        # Calcolo F1 veloce per ottimizzazione
        # Usiamo 'weighted' o binary pos_label=1. Qui assumiamo '1' come classe positiva (Attack)
        report = classification_report(y_true, preds, output_dict=True, zero_division=0)
        
        # F1-Score della classe '1' (Attack) è la metrica target solitamente
        current_f1 = report['1']['f1-score'] if '1' in report else 0.0
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_th = th
            best_preds = preds
            
    return best_f1, best_th, best_preds

def main():
    print(f"Caricamento dati da {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
        # Verifica veloce delle colonne
        required_cols = ['True_Label'] + [f'LogProb_Pos{i}' for i in range(5)]
        # Gestione caso in cui il CSV abbia nomi colonne diversi o manchino
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"❌ Errore: Colonne mancanti nel CSV: {missing}")
            return
    except FileNotFoundError:
        print(f"❌ File '{INPUT_FILE}' non trovato. Assicurati che sia nella stessa cartella.")
        return

    print("\n🔍 RICERCA BEST THRESHOLD PER STRATEGIA")
    
    strategies = ['majority', 'at_least_1', 'at_least_2', 'strict_all']
    results = {}

    # Fase 1: Ottimizzazione
    for strat in strategies:
        f1, th, preds = run_voting_strategy_optimization(df, strat)
        results[strat] = {'f1': f1, 'th': th, 'preds': preds}
        print(f"Strategia {strat.upper():<12} -> Miglior F1 (Attack): {f1:.4f} (Soglia Ottimale: {th:.4f})")

    # Trova il vincitore
    best_strat = max(results, key=lambda k: results[k]['f1'])
    
    print("\n" + "#"*60)
    print(" RISULTATI DETTAGLIATI ")
    print("#"*60)

    # Fase 2: Stampa Report Completi
    for strat in strategies:
        res = results[strat]
        print_custom_report(
            df['True_Label'], 
            res['preds'], 
            f"STRATEGIA: {strat.upper()} (Soglia Ottimale: {res['th']:.4f})"
        )

    print(f"\n🏆 VINCITORE ASSOLUTO: {best_strat.upper()} con F1-Score (Attack): {results[best_strat]['f1']:.4f}")

if __name__ == "__main__":
    main()