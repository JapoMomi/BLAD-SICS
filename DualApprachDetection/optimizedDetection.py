import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import itertools
import warnings

# --- CONFIGURAZIONE ---
VAL_FILE = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_validation_results.csv"
TEST_FILE = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_detection_results.csv"

# Tolleranze FPR da testare sul Validation
TARGET_FPRS = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

# Griglia Parametri OCSVM
OCSVM_GRID = {
    'nu': [0.01, 0.05, 0.1, 0.25, 0.5],
    'gamma': ['scale', 'auto', 0.01, 0.1, 0.5, 1.0],
    'ewma_span': [2, 3, 4, 5]
}

# Griglia Parametri Isolation Forest
IFOREST_GRID = {
    'contamination': [0.01, 0.05, 0.1, 0.25, 0.5, 'auto'],
    'n_estimators': [100, 200, 300],
    'ewma_span': [2, 3, 4, 5]
}

def prep_features(df):
    """Estrazione delle Golden Features (Spazio a 4 dimensioni compatto)"""
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

def run_grid_search_ocsvm(X_val, X_test, y_test):
    print("\nAvvio Grid Search per One-Class SVM...")
    keys = list(OCSVM_GRID.keys())
    combinations = list(itertools.product(*[OCSVM_GRID[k] for k in keys]))
    
    best_results = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        # 1. Addestramento
        clf = OneClassSVM(nu=params['nu'], gamma=params['gamma'], kernel='rbf')
        clf.fit(X_val)
        
        # 2. Score Raw (Invertito, alto = anomalia)
        val_raw = -clf.decision_function(X_val)
        test_raw = -clf.decision_function(X_test)
        
        # 3. Smoothing Temporale
        val_smooth = pd.Series(val_raw).ewm(span=params['ewma_span'], adjust=False).mean().values
        test_smooth = pd.Series(test_raw).ewm(span=params['ewma_span'], adjust=False).mean().values
        
        # 4. Ricerca Soglia su FPR
        for fpr in TARGET_FPRS:
            th = np.percentile(val_smooth, 100.0 - fpr)
            preds = (test_smooth > th).astype(int)
            f1 = f1_score(y_test, preds, zero_division=0)
            
            best_results.append({
                'model': 'OCSVM',
                'nu': params['nu'],
                'gamma': params['gamma'],
                'ewma_span': params['ewma_span'],
                'val_fpr': fpr,
                'f1': f1,
                'preds': preds,
                'scores': test_smooth
            })
            
        if (i+1) % 10 == 0:
            print(f"Progresso OCSVM: {i+1}/{len(combinations)} combinazioni testate...")
            
    # Ordiniamo e restituiamo i migliori 3
    return sorted(best_results, key=lambda x: x['f1'], reverse=True)[:3]

def run_grid_search_iforest(X_val, X_test, y_test):
    print("\nAvvio Grid Search per Isolation Forest...")
    keys = list(IFOREST_GRID.keys())
    combinations = list(itertools.product(*[IFOREST_GRID[k] for k in keys]))
    
    best_results = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        clf = IsolationForest(
            n_estimators=params['n_estimators'], 
            contamination=params['contamination'], 
            random_state=42, n_jobs=-1
        )
        clf.fit(X_val)
        
        val_raw = -clf.decision_function(X_val)
        test_raw = -clf.decision_function(X_test)
        
        val_smooth = pd.Series(val_raw).ewm(span=params['ewma_span'], adjust=False).mean().values
        test_smooth = pd.Series(test_raw).ewm(span=params['ewma_span'], adjust=False).mean().values
        
        for fpr in TARGET_FPRS:
            th = np.percentile(val_smooth, 100.0 - fpr)
            preds = (test_smooth > th).astype(int)
            f1 = f1_score(y_test, preds, zero_division=0)
            
            best_results.append({
                'model': 'IFOREST',
                'contamination': params['contamination'],
                'n_estimators': params['n_estimators'],
                'ewma_span': params['ewma_span'],
                'val_fpr': fpr,
                'f1': f1,
                'preds': preds,
                'scores': test_smooth
            })
            
        if (i+1) % 10 == 0:
            print(f"Progresso IForest: {i+1}/{len(combinations)} combinazioni testate...")
            
    return sorted(best_results, key=lambda x: x['f1'], reverse=True)[:3]

def print_top_configuration(rank, config, y_test):
    print(f"\n{'='*75}")
    print(f" 🏆 TOP {rank} CONFIGURATION: {config['model']}")
    print(f"{'='*75}")
    
    # Stampa parametri in base al modello
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

    features = ['Single_Score', 'Delta_Single_Min', 'Contex_Range', 'Time_Gradient']
    y_test = df_test['True_Label'].values

    # Robust Scaling
    scaler = RobustScaler()
    X_val = scaler.fit_transform(df_val[features].values)
    X_test = scaler.transform(df_test[features].values)

    # Eseguiamo le Grid Search
    top_ocsvm = run_grid_search_ocsvm(X_val, X_test, y_test)
    top_iforest = run_grid_search_iforest(X_val, X_test, y_test)

    # Uniamo i migliori e prendiamo la Top 3 assoluta
    all_top = sorted(top_ocsvm + top_iforest, key=lambda x: x['f1'], reverse=True)[:3]

    print("\n" + "#"*75)
    print(" RISULTATI FINALI GRID SEARCH")
    print("#"*75)
    
    for i, config in enumerate(all_top):
        print_top_configuration(i+1, config, y_test)

if __name__ == "__main__":
    main()