import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, classification_report

# Load the results you just saved
df = pd.read_csv("/home/spritz/storage/disk0/Master_Thesis/Stuff/detection_results_sliding_window.csv")

y_true = df['Label_True']
y_scores = -df['Anomaly_Score'] # Flip sign because ROC expects higher score = anomaly

# Calculate ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Calculate Youden's J statistic for each threshold
J = tpr - fpr
ix = np.argmax(J)
best_thresh_flipped = thresholds[ix]
best_threshold = -best_thresh_flipped # Flip sign back

print(f"Best Threshold: {best_threshold:.4f}")

# Apply new threshold
y_pred_new = (df['Anomaly_Score'] < best_threshold).astype(int)

print("\nNew Classification Report:")
print(classification_report(y_true, y_pred_new, target_names=["Benign", "Attack"]))