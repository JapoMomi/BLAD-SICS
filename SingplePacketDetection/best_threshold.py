import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, classification_report, roc_auc_score

# Carica i risultati esistenti
df = pd.read_csv("/home/spritz/storage/disk0/Master_Thesis/SingplePacketDetection/detection_results_sliding_window.csv")
y_true = df['Label_True']

# Testiamo tutte e 3 le metriche per vedere quale è la migliore
metrics = ['Score_Mean', 'Score_Min', 'Score_Median']

print(f"{'Metric':<15} | {'AUC':<10} | {'Best Threshold':<15}")
print("-" * 50)

best_metric_name = ""
best_auc = 0

for metric in metrics:
    # Flip score because usually lower score = anomaly
    y_scores = -df[metric]
    
    # Calcola AUC
    auc = roc_auc_score(y_true, y_scores)
    
    # Calcola soglia ottimale (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = -thresholds[ix] # Flip back
    
    print(f"{metric:<15} | {auc:.4f}     | {best_thresh:.4f}")
    
    if auc > best_auc:
        best_auc = auc
        best_metric_name = metric

print("-" * 50)
print(f"WINNER: {best_metric_name} with AUC {best_auc:.4f}")

# --- REPORT CON LA METRICA MIGLIORE ---
print(f"\n--- Classification Report using {best_metric_name} ---")
best_y_scores = -df[best_metric_name]
fpr, tpr, thresholds = roc_curve(y_true, best_y_scores)
ix = np.argmax(tpr - fpr)
optimal_threshold = -thresholds[ix]

y_pred = (df[best_metric_name] < optimal_threshold).astype(int)
print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"]))

# Matrice confusione rapida
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
print(f"TP: {cm[1][1]} (Attacks caught) | FN: {cm[1][0]} (Attacks missed)")
print(f"FP: {cm[0][1]} (False Alarms)  | TN: {cm[0][0]} (Benign ok)")