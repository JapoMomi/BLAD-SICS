import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix, average_precision_score, precision_recall_curve, roc_curve
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURAZIONE ---
VAL_FILE = "/home/spritz/storage/disk0/Master_Thesis/DualModelDetection/dual_model_validation_results.csv"
TEST_FILE = "/home/spritz/storage/disk0/Master_Thesis/DualModelDetection/dual_model_detection_results.csv"

# --- PARAMETRI PER LE METRICHE AVANZATE (REVIEWER) ---
TEST_SET_HOURS = 11.7     # <-- AGGIORNA QUESTO VALORE CON LE ORE REALI DEL TEST SET
TARGET_FPR_MATCH = 0.0112 # FPR 1.12% dal Single Packet (Min) Model

def prep_features(df):
    """Estrazione delle Golden Features (Ora include Min_Single_Score)"""
    context_cols = [c for c in df.columns if 'Ctx_Pos' in c]
    
    # Pulizia Base
    df[context_cols] = df[context_cols].apply(lambda row: row.fillna(row.mean()), axis=1)
    df['Single_Score'] = df['Single_Score'].fillna(df['Single_Score'].mean())
    
    if 'Min_Single_Score' in df.columns:
        df['Min_Single_Score'] = df['Min_Single_Score'].fillna(df['Min_Single_Score'].mean())
    
    # Calcolo Features Derivate
    df['Min_Context'] = df[context_cols].min(axis=1)
    df['Mean_Context'] = df[context_cols].mean(axis=1)
    df['Max_Context'] = df[context_cols].max(axis=1)
    df['Std_Context'] = df[context_cols].std(axis=1).fillna(0)
    
    df['Delta_Single_Min'] = df['Single_Score'] - df['Min_Context']
    df['Delta_Single_Mean'] = df['Single_Score'] - df['Mean_Context']

    df['Contex_Range'] = df['Max_Context'] - df['Min_Context']

    if 'Ctx_Pos0' in df.columns and 'Ctx_Pos4' in df.columns:
        df['Time_Gradient'] = df['Ctx_Pos4'] - df['Ctx_Pos0']
    else:
        df['Time_Gradient'] = 0.0
        
    return df

def run_best_ocsvm(X_val, X_test, y_test):
    print("\nEsecuzione OCSVM con i migliori parametri (nu=0.05, gamma=0.05, EWMA=3, FPR=1.0%)...")
    
    clf = OneClassSVM(nu=0.05, gamma=0.05, kernel='rbf')
    clf.fit(X_val)
    
    val_raw = -clf.decision_function(X_val)
    test_raw = -clf.decision_function(X_test)
    
    val_smooth = pd.Series(val_raw).ewm(span=3, adjust=False).mean().values
    test_smooth = pd.Series(test_raw).ewm(span=3, adjust=False).mean().values
    
    th = np.percentile(val_smooth, 100.0 - 1.0)
    preds = (test_smooth > th).astype(int)
    
    return {
        'model': 'OCSVM',
        'nu': 0.05, 'gamma': 0.05, 'ewma_span': 3, 'val_fpr': 1.0,
        'f1': f1_score(y_test, preds, zero_division=0),
        'preds': preds,
        'scores': test_smooth
    }

def run_best_iforest(X_val, X_test, y_test):
    print("\nEsecuzione Isolation Forest con i migliori parametri (cont=0.01, n_est=100, EWMA=2, FPR=0.5%)...")
    
    clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
    clf.fit(X_val)
    
    val_raw = -clf.decision_function(X_val)
    test_raw = -clf.decision_function(X_test)
    
    val_smooth = pd.Series(val_raw).ewm(span=2, adjust=False).mean().values
    test_smooth = pd.Series(test_raw).ewm(span=2, adjust=False).mean().values
    
    th = np.percentile(val_smooth, 100.0 - 0.5)
    preds = (test_smooth > th).astype(int)
    
    return {
        'model': 'IFOREST',
        'contamination': 0.01, 'n_estimators': 100, 'ewma_span': 2, 'val_fpr': 0.5,
        'f1': f1_score(y_test, preds, zero_division=0),
        'preds': preds,
        'scores': test_smooth
    }

def print_top_configuration(model_title, config, y_test):
    print(f"\n{'='*75}")
    print(f" 🏆 TOP CONFIGURATION: {model_title}")
    print(f"{'='*75}")
    
    if config['model'] == 'OCSVM':
        print(f"Parametri: nu={config['nu']} | gamma={config['gamma']} | EWMA={config['ewma_span']} | Target FPR={config['val_fpr']}%")
    else:
        print(f"Parametri: contamination={config['contamination']} | n_estimators={config['n_estimators']} | EWMA={config['ewma_span']} | Target FPR={config['val_fpr']}%")
        
    print(classification_report(y_test, config['preds'], digits=4, target_names=["Benign", "Attack"]))
    cm = confusion_matrix(y_test, config['preds'])
    print(f"Confusion Matrix:\n[TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}]\n[FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}]")
    
    try:
        print(f"ROC AUC: {roc_auc_score(y_test, config['scores']):.4f}")
    except ValueError:
        pass

def advanced_evaluation_for_reviewers(y_true, y_probs, model_name):
    """Calcola PR-AUC, Matched FPR performance, Alarms/Hour e salva la curva PR."""
    print(f"\n{'-'*75}")
    print(f" ADVANCED METRICS FOR REVIEWER (CONCERNS 1 & 3): {model_name}")
    print(f"{'-'*75}")

    # 1. PR-AUC
    pr_auc = average_precision_score(y_true, y_probs)
    print(f"PR-AUC: {pr_auc:.4f}")

    # 2. Performance at Matched FPR
    fpr_curve, tpr_curve, thresholds_roc = roc_curve(y_true, y_probs)
    idx = np.argmin(np.abs(fpr_curve - TARGET_FPR_MATCH))
    matched_th = thresholds_roc[idx]
    actual_fpr = fpr_curve[idx]

    matched_preds = (y_probs > matched_th).astype(int)
    cm = confusion_matrix(y_true, matched_preds)
    
    # Generate the full classification report
    report = classification_report(y_true, matched_preds, digits=4, target_names=["Benign", "Attack"], zero_division=0)
    print(f"\n--- Performance at Matched FPR ---")
    print(f"Target FPR: {TARGET_FPR_MATCH*100:.2f}% | Actual FPR Achieved: {actual_fpr*100:.2f}%")
    print(f"Matched Threshold: {matched_th:.4f}")
    # Print the full table instead of just the F1-score
    print("\nClassification Report:")
    print(report)
    print(f"Confusion Matrix:\n[TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}]\n[FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}]")

    # 3. Estimated Alarms per Hour
    fp = cm[0][1]
    alarms_per_hour = fp / TEST_SET_HOURS
    print(f"\n--- Operational Impact ---")
    print(f"Estimated Alarms/Hour (Based on {TEST_SET_HOURS} hours): {alarms_per_hour:.2f}")

    # 4. Save PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name.replace("_", " ")} (PR-AUC = {pr_auc:.4f})', color='darkred', lw=2)
    plt.xlabel('Recall (True Positive Rate)', fontsize=12)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
    plt.title(f'Precision-Recall Curve: {model_name.replace("_", " ")}', fontsize=14)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_filename = f"/home/spritz/storage/disk0/Master_Thesis/ReviewerImprovements/PR_Curve_{model_name}.png"
    plt.savefig(plot_filename, format='png', dpi=300)
    print(f"\n[+] Precision-Recall curve successfully saved as '{plot_filename}'.")
def main():
    print("Caricamento Dataset...")
    try:
        df_val = pd.read_csv(VAL_FILE)
        df_test = pd.read_csv(TEST_FILE)
    except FileNotFoundError as e:
        print(f"Errore: {e}")
        return

    if 'True_Label' in df_val.columns and df_val['True_Label'].sum() > 0:
        df_val = df_val[df_val['True_Label'] == 0].copy()

    df_val = prep_features(df_val)
    df_test = prep_features(df_test)

    features = ['Single_Score', 'Min_Single_Score', 'Delta_Single_Min', 'Contex_Range', 'Time_Gradient']
    
    for f in features:
        if f not in df_val.columns:
            print(f"⚠️ Attenzione: la feature '{f}' non è presente nel dataset. Controlla il CSV.")
            return

    y_test = df_test['True_Label'].values

    scaler = RobustScaler()
    X_val = scaler.fit_transform(df_val[features].values)
    X_test = scaler.transform(df_test[features].values)

    # 1. Esecuzione dei Modelli con i migliori parametri
    best_ocsvm = run_best_ocsvm(X_val, X_test, y_test)
    best_iforest = run_best_iforest(X_val, X_test, y_test)

    # 2. Stampa dei Report Classici
    print("\n" + "#"*75)
    print(" RISULTATI FINALI: VINCITORI ASSOLUTI")
    print("#"*75)
    print_top_configuration("ONE-CLASS SVM", best_ocsvm, y_test)
    print_top_configuration("ISOLATION FOREST", best_iforest, y_test)

    # 3. Metriche Avanzate e Grafici PR (Richiesti dal Reviewer)
    print("\n" + "#"*75)
    print(" GENERAZIONE METRICHE AVANZATE E GRAFICI (CONCERNS 1 & 3)")
    print("#"*75)
    advanced_evaluation_for_reviewers(y_test, best_ocsvm['scores'], model_name="Dual_Model_OCSVM")
    advanced_evaluation_for_reviewers(y_test, best_iforest['scores'], model_name="Dual_Model_IForest")

if __name__ == "__main__":
    main()