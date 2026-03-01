import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

# --- CONFIGURAZIONE ---
INPUT_FILE = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/sequenceDetection_mean+min_Threshold/detection_detailed_results.csv"
OPTIMIZATION_TARGET = "f1"  # "f1" o "recall"

def print_custom_report(y_true, y_pred, title, auc_score=None):
    """
    Stampa il report formattato esattamente come richiesto.
    """
    print(f"\n--- {title} ---")
    
    # 1. Classification Report standard
    report = classification_report(y_true, y_pred, target_names=["Benign", "Attack"], digits=2)
    print(report)
    
    # 2. Confusion Matrix personalizzata
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"Confusion Matrix:")
    print(f"TP:    {tp} | FN:    {fn}")
    print(f"FP:     {fp} | TN:        {tn}")
    
    # 3. AUC (Se non passata, la calcola sulla predizione binaria)
    if auc_score is None:
        auc_score = roc_auc_score(y_true, y_pred)
        
    print(f"AUC: {auc_score:.4f}")
    print("#######################################################################################")

def find_best_single_threshold(y_true, scores, score_name):
    """
    Trova la soglia migliore per una singola metrica (es. solo Media o solo Minimo).
    """
    print(f"\n[Ottimizzazione] Cerco la miglior soglia per: {score_name}...")
    
    # Range di ricerca basato sui percentili (dal 0.1% al 50%)
    candidates = np.unique(np.percentile(scores, np.linspace(0.1, 50, 100)))
    
    best_f1 = -1
    best_thresh = 0
    
    for t in candidates:
        # Predizione: 1 se score < t (Anomalia)
        pred = (scores < t).astype(int)
        current_f1 = f1_score(y_true, pred, pos_label=1)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = t
            
    return best_thresh

def analyze_all_strategies():
    print(f"--- Loading results from {INPUT_FILE} ---")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Errore: File non trovato.")
        return

    y_true = df['Label'].values
    
    # Calcolo Min_Score se manca
    if 'Min_Score' not in df.columns:
        score_cols = [c for c in df.columns if c.startswith('Score_P')]
        if score_cols:
            df['Min_Score'] = df[score_cols].min(axis=1)
        else:
            print("ERRORE: Colonne Score_P... non trovate.")
            return

    avg_scores = df['Avg_Score'].values
    min_scores = df['Min_Score'].values

    # ==========================================
    # 1. STRATEGIA: SOLO MEDIA (MEAN ONLY)
    # ==========================================
    t_mean_opt = find_best_single_threshold(y_true, avg_scores, "Mean Only")
    pred_mean = (avg_scores < t_mean_opt).astype(int)
    print_custom_report(y_true, pred_mean, "1. SOLO MEAN (Avg_Score)", roc_auc_score(y_true, -avg_scores))

    # ==========================================
    # 2. STRATEGIA: SOLO MINIMO (MIN ONLY)
    # ==========================================
    t_min_opt = find_best_single_threshold(y_true, min_scores, "Min Only")
    pred_min = (min_scores < t_min_opt).astype(int)
    print_custom_report(y_true, pred_min, "2. SOLO MIN (Min_Score)", roc_auc_score(y_true, -min_scores))

    # ==============================================================================
    # IBRIDO "NAIVE"
    # Usa le soglie trovate sopra (t_mean_opt e t_min_opt) e le unisce con OR
    # ==============================================================================
    pred_naive = ((avg_scores < t_mean_opt) | (min_scores < t_min_opt)).astype(int)
    print_custom_report(y_true, pred_naive, "3. IBRIDO NAIVE (Unione soglie singole)")

    # ==========================================
    # 3. STRATEGIA: IBRIDO OTTIMIZZATO (GRID SEARCH)
    # ==========================================
    print(f"\n[Ottimizzazione] Cerco soglie ibride ottimali (Avg OR Min)...")
    
    # Grid Search ridotta per velocità
    avg_candidates = np.unique(np.percentile(avg_scores, np.linspace(0.1, 40, 50)))
    min_candidates = np.unique(np.percentile(min_scores, np.linspace(0.1, 30, 50)))
    
    best_hybrid_f1 = -1
    best_h_avg = 0
    best_h_min = 0
    
    for t_a in avg_candidates:
        for t_m in min_candidates:
            # Logica OR
            pred = ((avg_scores < t_a) | (min_scores < t_m)).astype(int)
            f1 = f1_score(y_true, pred, pos_label=1)
            
            if f1 > best_hybrid_f1:
                best_hybrid_f1 = f1
                best_h_avg = t_a
                best_h_min = t_m

    pred_hybrid = ((avg_scores < best_h_avg) | (min_scores < best_h_min)).astype(int)
    
    # Per l'ibrido l'AUC sui raw scores è complessa, usiamo l'AUC della decisione binaria
    auc_hybrid = roc_auc_score(y_true, pred_hybrid)
    
    print_custom_report(y_true, pred_hybrid, "Simulazione: IBRIDA (Mean OR Min)", auc_score=auc_hybrid)
    
    print("\n--- Riepilogo Soglie Scelte ---")
    print(f"Mean Only Threshold: {t_mean_opt:.4f}")
    print(f"Min Only Threshold:  {t_min_opt:.4f}")
    print(f"Hybrid Thresholds:   Avg < {best_h_avg:.4f} OR Min < {best_h_min:.4f}")

if __name__ == "__main__":
    analyze_all_strategies()