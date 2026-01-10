import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt # Opzionale, per grafici se servono

# --- CONFIGURAZIONE ---
INPUT_FILE = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/detection_detailed_results.csv"

def analyze_results():
    print(f"--- Loading results from {INPUT_FILE} ---")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Errore: File non trovato. Esegui prima lo script di detection!")
        return

    y_true = df['Label']
    
    # 1. Creiamo metriche derivate
    # Avg_Score esiste già. Calcoliamo il MINIMO tra i 5 pacchetti per vedere se funziona meglio.
    score_cols = [c for c in df.columns if c.startswith('Score_P')]
    if score_cols:
        df['Min_Score'] = df[score_cols].min(axis=1)
        metrics = ['Avg_Score', 'Min_Score']
    else:
        metrics = ['Avg_Score']

    print(f"\nConfronto Metriche (Score più alti = Benigni, più bassi = Attacchi):")
    print(f"{'Metric':<15} | {'AUC':<10} | {'Best Threshold':<15}")
    print("-" * 50)

    best_metric_name = ""
    best_auc = -1
    best_thresh_val = 0

    # 2. Loop su ogni metrica (Media vs Min)
    for metric in metrics:
        # IMPORTANTE: Nel nostro codice, Score basso (es -10) = Attacco (Class 1).
        # Score alto (es -0.1) = Benigno (Class 0).
        # La funzione roc_curve si aspetta che "Score Alto" = "Classe 1".
        # Quindi dobbiamo INVERTIRE il segno dello score per il calcolo ROC.
        y_scores_for_roc = -df[metric] 
        
        # Calcolo AUC
        auc = roc_auc_score(y_true, y_scores_for_roc)
        
        # Calcolo Soglia Ottimale (Youden's J)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores_for_roc)
        J = tpr - fpr
        ix = np.argmax(J)
        
        # La soglia restituita da roc_curve è sul valore invertito. La giriamo di nuovo.
        best_thresh = -thresholds[ix] 
        
        print(f"{metric:<15} | {auc:.4f}     | {best_thresh:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_metric_name = metric
            best_thresh_val = best_thresh

    print("-" * 50)
    print(f"VINCITORE: {best_metric_name} con AUC {best_auc:.4f}")
    print(f"Soglia Ottimale suggerita: {best_thresh_val:.4f}")

    # 3. Report Dettagliato sul Vincitore
    print(f"\n--- Simulation using Best Threshold ({best_thresh_val:.4f}) on {best_metric_name} ---")
    
    # Applichiamo la soglia: Se Score < Soglia -> Attacco (1), altrimenti Benigno (0)
    y_pred = (df[best_metric_name] < best_thresh_val).astype(int)
    
    print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"]))
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"Confusion Matrix:")
    print(f"TP (Attacchi Presi):    {tp} | FN (Attacchi Persi):    {fn}")
    print(f"FP (Falsi Allarmi):     {fp} | TN (Benigni OK):        {tn}")
    
    fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"\nFalse Positive Rate (FPR): {fpr_rate*100:.2f}%")

if __name__ == "__main__":
    analyze_results()