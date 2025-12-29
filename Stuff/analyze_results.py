import pandas as pd
import matplotlib.pyplot as plt

# Load your results
df = pd.read_csv("/home/spritz/storage/disk0/Master_Thesis/Stuff/detection_results_sliding_window.csv")

# Filter for False Positives (True Label = Benign/0, but Predicted = Attack/1)
fp_df = df[(df['Label_True'] == 0) & (df['Label_Pred'] == 1)]

print(f"Analyzing {len(fp_df)} False Positives...")
print(fp_df[['Score_Mean', 'Score_Min', 'Score_Max']].describe())

# Check the difference
# If (Mean - Min) is LARGE, it means 1 bad byte dragged the score down.
# If (Mean - Min) is SMALL, it means the whole packet was considered 'bad'.
df['Diff'] = df['Score_Mean'] - df['Score_Min']
print("\nAverage Difference between Mean and Min on False Positives:")
print(df[df['Label_Pred'] == 1]['Diff'].mean())