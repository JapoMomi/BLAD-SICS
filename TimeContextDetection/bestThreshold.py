import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# --- CONFIGURAZIONE ---
INPUT_FILE = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/detection_sliding_results.csv"

def print_custom_report(y_true, y_pred, strategy_name):
    """
    Stampa il report formattato esattamente come richiesto dall'utente.
    """
    # Calcolo metriche base
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"STRATEGIA: {strategy_name.upper()}")
    print(f"{'='*60}")
    
    # Header
    print(f"{'':<14} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}")
    print("")
    
    # Class 0 (Benign)
    r0 = report_dict['0']
    print(f"{'Benign':<14} {r0['precision']:>9.2f} {r0['recall']:>9.2f} {r0['f1-score']:>9.2f} {r0['support']:>9}")
    
    # Class 1 (Attack)
    r1 = report_dict['1']
    print(f"{'Attack':<14} {r1['precision']:>9.2f} {r1['recall']:>9.2f} {r1['f1-score']:>9.2f} {r1['support']:>9}")
    print("")
    
    # Totals
    print(f"{'accuracy':<14} {'':>9} {'':>9} {acc:>9.2f} {report_dict['macro avg']['support']:>9}")
    
    rm = report_dict['macro avg']
    print(f"{'macro avg':<14} {rm['precision']:>9.2f} {rm['recall']:>9.2f} {rm['f1-score']:>9.2f} {rm['support']:>9}")
    
    rw = report_dict['weighted avg']
    print(f"{'weighted avg':<14} {rw['precision']:>9.2f} {rw['recall']:>9.2f} {rw['f1-score']:>9.2f} {rw['support']:>9}")
    
    print(f"\nConfusion Matrix: [TP: {tp} | FN: {fn}]")
    print(f"                  [FP: {fp} | TN: {tn}]")
    
    return r1['f1-score'] # Ritorna f1 su classe 1 per il ranking

def apply_strategy(df, threshold, strategy):
    """Applica la logica di voting per decidere se un pacchetto è anomalo"""
    
    # Seleziona solo le colonne degli score
    score_cols = ['Score_Pos0', 'Score_Pos1', 'Score_Pos2', 'Score_Pos3', 'Score_Pos4']
    scores = df[score_cols]
    
    # Maschera booleana: Dove lo score supera la soglia?
    exceeds = scores > threshold
    
    if strategy == 'majority':
        # Conta quante volte supera la soglia
        count_exceed = exceeds.sum(axis=1)
        # Conta quanti valori validi (non-NaN) ci sono per riga
        count_valid = scores.notna().sum(axis=1)
        # Se supera la metà dei test validi -> Anomalous
        # Gestisce il caso count_valid=0 restituendo False
        return (count_exceed > (count_valid / 2)).astype(int)
        
    elif strategy == 'at_least_1':
        # Se almeno uno score supera la soglia -> Anomalous
        return exceeds.any(axis=1).astype(int)
        
    elif strategy == 'strict_all':
        # Se TUTTI gli score validi superano la soglia (e c'è almeno uno score valido)
        # Trucco: fillna(True) su exceeds non va bene perché NaN non deve contare come True.
        # Logica: (exceeds.sum == notna.sum) AND (notna.sum > 0)
        all_match = (exceeds.sum(axis=1) == scores.notna().sum(axis=1))
        has_data = scores.notna().sum(axis=1) > 0
        return (all_match & has_data).astype(int)
        
    elif strategy == 'average':
        # Media degli score validi > threshold
        avg_scores = scores.mean(axis=1) # Pandas mean ignora NaN di default
        return (avg_scores > threshold).astype(int)

def main():
    print(f"Caricamento dati da {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("❌ File non trovato. Esegui prima detection.py")
        return

    y_true = df['True_Label'].values
    
    # Calcoliamo il range di threshold da testare
    # Prendiamo min e max di tutti gli score nel dataframe
    score_cols = ['Score_Pos0', 'Score_Pos1', 'Score_Pos2', 'Score_Pos3', 'Score_Pos4']
    all_scores = df[score_cols].values.flatten()
    all_scores = all_scores[~np.isnan(all_scores)] # Rimuovi NaN
    
    min_th = np.percentile(all_scores, 1)  # Evita outlier estremi
    max_th = np.percentile(all_scores, 99)
    
    # Genera 100 possibili threshold
    thresholds = np.linspace(min_th, max_th, 100)
    
    strategies = ['majority', 'at_least_1', 'strict_all', 'average']
    
    # Per ogni strategia, teniamo traccia del miglior risultato
    best_results = {} # {strategy: {'th': float, 'f1': float, 'preds': array}}

    print(f"🔍 Ricerca della Best Threshold testando {len(thresholds)} valori per 4 strategie...")
    
    for strat in strategies:
        best_f1 = -1
        best_th = 0
        best_preds = None
        
        # Iterazione rapida per trovare la threshold migliore
        for th in thresholds:
            y_pred = apply_strategy(df, th, strat)
            # Calcoliamo solo F1 classe 1 velocemente
            f1 = classification_report(y_true, y_pred, output_dict=True)['1']['f1-score']
            
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
                best_preds = y_pred
        
        best_results[strat] = {
            'th': best_th,
            'f1': best_f1,
            'preds': best_preds
        }

    # --- STAMPA REPORT FINALI ---
    print("\n\n" + "#"*40)
    print(" RISULTATI DETTAGLIATI PER STRATEGIA (BEST THRESHOLD)")
    print("#"*40)

    results_summary = []

    for strat in strategies:
        res = best_results[strat]
        print(f"\n>>> Analisi Strategia: {strat.upper()} (Best Th: {res['th']:.4f})")
        f1_score_final = print_custom_report(y_true, res['preds'], strat)
        results_summary.append((strat, f1_score_final))

    # --- PROCLAMAZIONE VINCITORE ---
    results_summary.sort(key=lambda x: x[1], reverse=True)
    winner = results_summary[0]
    
    print("\n\n🏆 VINCITORE ASSOLUTO 🏆")
    print(f"La strategia migliore è: {winner[0].upper()} con F1-Score: {winner[1]:.2f}")
    print("Usa questa logica e la threshold indicata sopra nel sistema di produzione.")

if __name__ == "__main__":
    main()