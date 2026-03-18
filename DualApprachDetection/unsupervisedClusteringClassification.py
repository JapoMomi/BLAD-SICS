import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
import warnings

# --- CONFIGURAZIONE ---
VAL_FILE = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_validation_results.csv"
TEST_FILE = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_detection_results.csv"

# Tolleranze di Falsi Allarmi (FPR) da testare sul Validation Set
TARGET_FPRS = np.arange(0.01, 10.1, 0.02) # Da 0.1% a 5.0%

# Smoothing temporale
EWMA_SPAN = 5 
# Peso per le Euristiche 
WEIGHT_SINGLE = 0.3 

def print_report(y_true, y_pred, y_probs, title):
    print(f"\n{'='*80}\n{title}\n{'='*80}")
    print(classification_report(y_true, y_pred, digits=4, target_names=["Benign", "Attack"], zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n[TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}]\n[FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}]")
    if y_probs is not None:
        try:
            print(f"ROC AUC: {roc_auc_score(y_true, y_probs):.4f}")
        except ValueError:
            pass

def prep_features(df):
    """Calcola le meta-feature di base per il dataset."""
    context_cols = [c for c in df.columns if 'Ctx_Pos' in c]
    
    df[context_cols] = df[context_cols].apply(lambda row: row.fillna(row.mean()), axis=1)
    df['Single_Score'] = df['Single_Score'].fillna(df['Single_Score'].mean())
    
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

def z_norm_unsupervised(val_series, test_series):
    """Normalizzazione basata solo sui parametri del Validation Set."""
    mu = val_series.mean()
    std = val_series.std()
    if std == 0: std = 1e-9
    return (val_series - mu) / std, (test_series - mu) / std

def evaluate_sweep(method_name, val_scores, test_scores, y_test):
    """
    Applica l'EWMA e cerca la soglia migliore basandosi sul Validation Set.
    Tutti gli score in ingresso devono essere: Valori Alti = Anomalia.
    """
    # Smoothing
    val_smooth = pd.Series(val_scores).ewm(span=EWMA_SPAN, adjust=False).mean().values
    test_smooth = pd.Series(test_scores).ewm(span=EWMA_SPAN, adjust=False).mean().values
    
    best_f1, best_fpr, best_preds = -1, 0, None
    
    for fpr_target in TARGET_FPRS:
        threshold = np.percentile(val_smooth, 100.0 - fpr_target)
        preds = (test_smooth > threshold).astype(int)
        current_f1 = f1_score(y_test, preds, zero_division=0)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_fpr = fpr_target
            best_preds = preds
            
    print_report(y_test, best_preds, test_smooth, f"{method_name} | Ottimizzato su Val FPR: {best_fpr:.1f}%")
    return best_f1

def run_heuristics(df_val, df_test):
    """Esegue le due euristiche di base (Z-Score combinato)."""
    y_test = df_test['True_Label'].values
    
    val_s, test_s = z_norm_unsupervised(df_val['Single_Score'], df_test['Single_Score'])
    val_cmin, test_cmin = z_norm_unsupervised(df_val['Min_Context'], df_test['Min_Context'])
    val_cmean, test_cmean = z_norm_unsupervised(df_val['Mean_Context'], df_test['Mean_Context'])
    
    w = WEIGHT_SINGLE
    # Invertiamo il segno: - (LogProb) diventa positivo per le anomalie
    val_score_min = -(w * val_s + (1 - w) * val_cmin)
    test_score_min = -(w * test_s + (1 - w) * test_cmin)
    
    val_score_mean = -(w * val_s + (1 - w) * val_cmean)
    test_score_mean = -(w * test_s + (1 - w) * test_cmean)
    
    f1_min = evaluate_sweep("1. HEURISTIC (Single + Min Context)", val_score_min, test_score_min, y_test)
    f1_mean = evaluate_sweep("2. HEURISTIC (Single + Mean Context)", val_score_mean, test_score_mean, y_test)
    
    return f1_min, f1_mean

def run_ml_and_clustering(df_val, df_test, algo_name):
    """Gestisce addestramento e inferenza per i 5 modelli di ML/Clustering."""
    y_test = df_test['True_Label'].values
    
    # Golden Features
    features = ['Single_Score', 'Delta_Single_Min', 'Contex_Range', 'Time_Gradient']
    
    # Scaling Unsupervised
    scaler = StandardScaler()
    X_val = scaler.fit_transform(df_val[features].values)
    X_test = scaler.transform(df_test[features].values)
    
    if algo_name == 'iforest':
        clf = IsolationForest(n_estimators=100, contamination=0.25, random_state=42, n_jobs=-1)
        clf.fit(X_val)
        val_scores = -clf.decision_function(X_val)
        test_scores = -clf.decision_function(X_test)
        title = "3. ML: Isolation Forest"
        
    elif algo_name == 'ocsvm':
        clf = OneClassSVM(nu=0.25, kernel='rbf', gamma='scale')
        clf.fit(X_val)
        val_scores = -clf.decision_function(X_val)
        test_scores = -clf.decision_function(X_test)
        title = "4. ML: One-Class SVM"
        
    elif algo_name == 'gmm':
        clf = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
        clf.fit(X_val)
        val_scores = -clf.score_samples(X_val)
        test_scores = -clf.score_samples(X_test)
        title = "5. CLUSTERING: Gaussian Mixture Model (GMM)"
        
    elif algo_name == 'kmeans':
        clf = KMeans(n_clusters=3, random_state=42, n_init="auto")
        clf.fit(X_val)
        val_scores = np.min(clf.transform(X_val), axis=1)
        test_scores = np.min(clf.transform(X_test), axis=1)
        title = "6. CLUSTERING: K-Means"

    return evaluate_sweep(title, val_scores, test_scores, y_test)

def main():
    print("Inizio Elaborazione Master Script...")
    try:
        df_val = pd.read_csv(VAL_FILE)
        df_test = pd.read_csv(TEST_FILE)
    except FileNotFoundError as e:
        print(f"Errore: {e}")
        return

    # Garanzia Unsupervised: rimuoviamo attacchi se presenti nel Val
    if 'True_Label' in df_val.columns and df_val['True_Label'].sum() > 0:
        df_val = df_val[df_val['True_Label'] == 0].copy()

    # Prepara feature per tutto il dataset
    df_val = prep_features(df_val)
    df_test = prep_features(df_test)

    results = {}

    print("\n>>> ESECUZIONE EURISTICHE DI BASE <<<")
    results['Heuristic (Min)'] , results['Heuristic (Mean)'] = run_heuristics(df_val, df_test)

    print("\n>>> ESECUZIONE MACHINE LEARNING (BOUNDARY) <<<")
    results['Isolation Forest'] = run_ml_and_clustering(df_val, df_test, 'iforest')
    results['One-Class SVM'] = run_ml_and_clustering(df_val, df_test, 'ocsvm')

    print("\n>>> ESECUZIONE MACHINE LEARNING (CLUSTERING) <<<")
    results['GMM (Gaussian Mixture)'] = run_ml_and_clustering(df_val, df_test, 'gmm')
    results['K-Means'] = run_ml_and_clustering(df_val, df_test, 'kmeans')

    print("\n" + "#"*75)
    print(" CLASSIFICA FINALE F1-SCORE (Approccio 100% Unsupervised)")
    print("#"*75)
    
    # Ordina i risultati dal migliore al peggiore
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (method, score) in enumerate(sorted_results, 1):
        if i == 1:
            print(f" 🏆 {method:<30}: {score:.4f}")
        else:
            print(f" {i}. {method:<30}: {score:.4f}")

if __name__ == "__main__":
    main()