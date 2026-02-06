import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score

# Carica i risultati salvati
df = pd.read_csv("/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_detection_results.csv")

def optimize_detection_mean(df, beta):
    """
    Ottimizzazione basata sulla MEDIA dei contesti (più conservativa sui FP).
    """
    
    # 1. AGGREGAZIONE (MEAN invece di MIN)
    context_cols = [c for c in df.columns if 'Ctx_Pos' in c]
    # Usiamo la media: riduce il rumore, ma rischia di nascondere picchi di anomalia
    df['Mean_Context'] = df[context_cols].mean(axis=1) 
    
    # 2. NORMALIZZAZIONE (Z-Score)
    # È fondamentale normalizzare perché la media avrà una scala diversa dal minimo
    
    # Single Score Norm
    s_mean = df['Single_Score'].mean()
    s_std = df['Single_Score'].std()
    df['S_Norm'] = (df['Single_Score'] - s_mean) / s_std
    
    # Context Score Norm (Sulla colonna Mean_Context)
    c_mean = df['Mean_Context'].mean()
    c_std = df['Mean_Context'].std()
    df['C_Norm'] = (df['Mean_Context'] - c_mean) / c_std
    
    # 3. RICERCA GRID SEARCH
    best_score = -1
    best_w = 0
    best_th = 0
    best_preds = None
    
    print(f"Ottimizzazione con MEDIA in corso (Target: F{beta}-Score)...")
    
    weights = np.arange(0.1, 1.0, 0.1)
    
    for w in weights:
        # Punteggio combinato pesato
        combined_scores = (w * df['S_Norm']) + ((1 - w) * df['C_Norm'])
        
        # Grid search sulla soglia
        thresholds = np.linspace(np.percentile(combined_scores, 1), np.percentile(combined_scores, 99), 100)
        
        for th in thresholds:
            preds = (combined_scores < th).astype(int)
            
            # Calcolo F-Beta
            current_score = fbeta_score(df['True_Label'], preds, beta=beta)
            
            if current_score > best_score:
                best_score = current_score
                best_w = w
                best_th = th
                best_preds = preds

    return best_w, best_th, best_preds

# --- ESECUZIONE ---
# beta=0.5 penalizza molto i FP (favorisce la Precision)
# Se vuoi bilanciare, usa beta=1.0
beta_val = 1

best_w, best_th, final_preds = optimize_detection_mean(df, beta=beta_val)

print(f"\n{'='*60}")
print(f"RISULTATI OTTIMIZZATI (Strategy: MEAN | Beta={beta_val})")
print(f"Miglior Peso SinglePacket: {best_w:.1f} (Context Mean: {1-best_w:.1f})")
print(f"Miglior Soglia Normalizzata: {best_th:.4f}")
print(f"{'='*60}")

print(classification_report(df['True_Label'], final_preds, digits=4))
cm = confusion_matrix(df['True_Label'], final_preds)
print(f"Confusion Matrix:\n[TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}]\n[FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}]")