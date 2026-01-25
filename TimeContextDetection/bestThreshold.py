import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURAZIONE ---
INPUT_FILE = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/detection_raw_logprobs.csv"

def print_custom_report(y_true, y_pred, title):
    """Funzione di servizio per stampare a schermo i risultati formattati"""
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"{'':<14} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}")
    r0 = report_dict['0']
    r1 = report_dict['1']
    print(f"{'Benign':<14} {r0['precision']:>9.2f} {r0['recall']:>9.2f} {r0['f1-score']:>9.2f} {r0['support']:>9}")
    print(f"{'Attack':<14} {r1['precision']:>9.2f} {r1['recall']:>9.2f} {r1['f1-score']:>9.2f} {r1['support']:>9}")
    print(f"\nConfusion Matrix: [TP: {tp} | FN: {fn}]")
    print(f"                  [FP: {fp} | TN: {tn}]")
    return r1['f1-score']

def run_voting_strategy(df, strategy_name):
    """
    Trova la soglia migliore per una specifica strategia di voting (es. majority).
    """
    y_true = df['True_Label'].values
    
    # Seleziona le 5 colonne contenenti le log-probabilità
    score_cols = [f'LogProb_Pos{i}' for i in range(5)]
    scores = df[score_cols]
    
    # Raccoglie tutti i punteggi validi per definire il range di ricerca della soglia
    vals = scores.values.flatten()
    vals = vals[~np.isnan(vals)]
    
    # Genera 100 possibili soglie: dal percentile 1 (valori molto negativi) 
    # al percentile 99 (valori vicini allo zero)
    thresholds = np.linspace(np.percentile(vals, 1), np.percentile(vals, 99), 100)
    
    best_f1 = -1
    best_th = None
    best_preds = None
    
    # Per ogni possibile soglia, simuliamo cosa accadrebbe
    for th in thresholds:
        # CONDIZIONE DI ANOMALIA: 
        # Siccome lavoriamo con log-probabilità negative, un pacchetto è anomalo
        # se la sua probabilità è MINORE della soglia (cioè è molto improbabile).
        is_anomalous = scores < th 
        
        # --- APPLICAZIONE DELLE STRATEGIE DI VOTING ---
        
        if strategy_name == 'majority':
            # Majority Voting:
            # exceeds.sum(axis=1) -> Conta quante volte su 5 il pacchetto è anomalo.
            # scores.notna().sum(axis=1) / 2 -> Calcola la metà dei test validi (es. 2.5 su 5).
            # Se i voti anomali superano la maggioranza, etichettiamo come 1 (Attacco).
            vote = (is_anomalous.sum(axis=1) > (scores.notna().sum(axis=1) / 2)).astype(int)
            
        elif strategy_name == 'at_least_1':
            # At Least 1:
            # is_anomalous.any(axis=1) -> Ritorna True se c'è ALMENO UN test anomalo sui 5.
            vote = is_anomalous.any(axis=1).astype(int)
            
        elif strategy_name == 'strict_all':
            # Strict (Unanimità):
            # Conta se il numero di test falliti è UGUALE al numero totale di test validi.
            all_match = (is_anomalous.sum(axis=1) == scores.notna().sum(axis=1))
            has_data = scores.notna().sum(axis=1) > 0 # Evita di votare su pacchetti senza dati
            vote = (all_match & has_data).astype(int)
        
        # Calcoliamo l'F1-Score ottenuto con questa soglia temporanea
        f1 = classification_report(y_true, vote, output_dict=True)['1']['f1-score']
        
        # Se questo risultato è migliore del precedente, lo salviamo come il "nuovo migliore"
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_preds = vote
            
    return best_f1, best_th, best_preds

def main():
    print(f"Caricamento dati da {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except:
        print("❌ File non trovato. Esegui prima detection.py")
        return

    print("\n🔍 RICERCA BEST THRESHOLD (Utilizzo Log-Likelihood Negativa)")
    
    # Lista delle strategie da testare
    strategies = ['majority', 'at_least_1', 'strict_all']
    results = {}

    # Eseguiamo la ricerca per ogni strategia
    for strat in strategies:
        f1, th, preds = run_voting_strategy(df, strat)
        results[strat] = {'f1': f1, 'th': th, 'preds': preds}
        print(f"Strategia {strat.upper():<12} -> Miglior F1: {f1:.3f} (con Soglia: {th:.4f})")

    # Identifichiamo la strategia che ha ottenuto l'F1-Score più alto in assoluto
    best_strat = max(results, key=lambda k: results[k]['f1'])
    
    # Stampiamo i report dettagliati per tutte le strategie testate
    print("\n" + "#"*60)
    print(" RISULTATI DETTAGLIATI DELLE STRATEGIE DI VOTING ")
    print("#"*60)

    for strat in strategies:
        print_custom_report(df['True_Label'], results[strat]['preds'], f"RISULTATI STRATEGIA: {strat.upper()} (Soglia: {results[strat]['th']:.4f})")

    print(f"\n🏆 VINCITORE: {best_strat.upper()} (F1-Score: {results[best_strat]['f1']:.3f})")

if __name__ == "__main__":
    main()