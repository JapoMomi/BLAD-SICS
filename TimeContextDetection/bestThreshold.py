import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# --- CONFIGURAZIONE ---
INPUT_FILE = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/detection_detailed_results.csv"

# Vuoi ottimizzare per F1-Score (bilanciato) o Recall (prendere tutti gli attacchi)?
# "f1" = Bilanciato (Consigliato)
# "recall" = Minimizzare i Falsi Negativi (rischio di più Falsi Positivi)
OPTIMIZATION_TARGET = "f1" 

def analyze_hybrid_thresholds():
    print(f"--- Loading results from {INPUT_FILE} ---")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Errore: File non trovato. Esegui prima lo script di detection!")
        return

    y_true = df['Label'].values
    
    # 1. Assicuriamoci di avere Avg_Score e Min_Score
    # Avg_Score c'è sicuro. Calcoliamo il MINIMO dai pacchetti se non c'è.
    if 'Min_Score' not in df.columns:
        score_cols = [c for c in df.columns if c.startswith('Score_P')]
        if score_cols:
            print("Calcolo colonna 'Min_Score' dai punteggi individuali...")
            df['Min_Score'] = df[score_cols].min(axis=1)
        else:
            print("ERRORE: Colonne Score_P... non trovate. Impossibile calcolare il Minimo.")
            return

    avg_scores = df['Avg_Score'].values
    min_scores = df['Min_Score'].values

    print(f"\n--- Avvio Grid Search Ibrida (Target: {OPTIMIZATION_TARGET.upper()}) ---")
    print("Cerchiamo la combinazione migliore: (Avg < T1) OR (Min < T2)")
    
    # 2. Definiamo lo spazio di ricerca (Grid) basato sui dati reali
    # Usiamo i percentili per non cercare valori impossibili.
    # Avg: Cerchiamo tra il valore minimo e il 30esimo percentile (zona di confine attacchi)
    avg_candidates = np.unique(np.percentile(avg_scores, np.linspace(0.1, 40, 50)))
    # Min: Cerchiamo tra il valore minimo e il 20esimo percentile
    min_candidates = np.unique(np.percentile(min_scores, np.linspace(0.1, 30, 50)))
    
    best_score = -1
    best_avg_thresh = 0
    best_min_thresh = 0
    
    total_combinations = len(avg_candidates) * len(min_candidates)
    print(f"Testando {total_combinations} combinazioni di soglie...")

    # 3. Grid Search Loop
    for t_avg in avg_candidates:
        for t_min in min_candidates:
            
            # LOGICA IBRIDA: Allarme se Media bassa OPPURE Minimo basso
            # Nota: Minimo basso cattura gli attacchi "cecchino" (1 pacchetto su 5)
            y_pred = ((avg_scores < t_avg) | (min_scores < t_min)).astype(int)
            
            if OPTIMIZATION_TARGET == "f1":
                # F1 Score sulla classe 1 (Attacco)
                score = f1_score(y_true, y_pred, pos_label=1)
            else:
                # Recall sulla classe 1
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                score = tp / (tp + fn) if (tp + fn) > 0 else 0
                # Penalità se FPR esplode (opzionale, per sicurezza)
                if (fp / (fp+tn)) > 0.3: score = 0 

            if score > best_score:
                best_score = score
                best_avg_thresh = t_avg
                best_min_thresh = t_min

    print("-" * 50)
    print(f"VINCITORE TROVATO!")
    print(f"Best {OPTIMIZATION_TARGET.upper()}: {best_score:.4f}")
    print(f"Soglia Avg: {best_avg_thresh:.4f}")
    print(f"Soglia Min: {best_min_thresh:.4f}")
    print("-" * 50)

    # 4. Report Dettagliato sul Vincitore
    print(f"\n--- Simulazione Finale con Soglie Ottimali ---")
    
    final_pred = ((avg_scores < best_avg_thresh) | (min_scores < best_min_thresh)).astype(int)
    
    print(classification_report(y_true, final_pred, target_names=["Benign", "Attack"]))
    
    cm = confusion_matrix(y_true, final_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"Confusion Matrix:")
    print(f"TP (Attacchi Presi):    {tp} | FN (Attacchi Persi):    {fn}")
    print(f"FP (Falsi Allarmi):     {fp} | TN (Benigni OK):        {tn}")
    
    fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"\nFalse Positive Rate (FPR): {fpr_rate*100:.2f}%")
    
    # 5. Salvataggio soglie su file per uso futuro
    with open("/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/best_thresholds_found.txt", "w") as f:
        f.write(f"AVG_THRESH={best_avg_thresh}\n")
        f.write(f"MIN_THRESH={best_min_thresh}\n")
    print("\nSoglie salvate in 'best_thresholds_found.txt'")

if __name__ == "__main__":
    analyze_hybrid_thresholds()