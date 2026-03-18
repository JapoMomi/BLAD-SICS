import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from scipy.ndimage import binary_closing, binary_opening
import warnings

# --- CONFIGURAZIONE ---
VAL_FILE = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_validation_results.csv"
TEST_FILE = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_detection_results.csv"

def print_report(y_true, y_pred, title):
    print(f"\n{'='*75}\n{title}\n{'='*75}")
    print(classification_report(y_true, y_pred, digits=4, target_names=["Benign", "Attack"]))
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n[TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}]\n[FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}]")
    print(f"F1-Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")

def prep_features(df):
    """Estrae le feature d'oro per la SVM"""
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

def apply_morphological_filters(preds, close_size=3, open_size=2):
    """Corregge i buchi negli attacchi e i falsi allarmi isolati"""
    str_close = np.ones(close_size, dtype=int)
    closed_preds = binary_closing(preds, structure=str_close).astype(int)
    
    str_open = np.ones(open_size, dtype=int)
    final_preds = binary_opening(closed_preds, structure=str_open).astype(int)
    return final_preds

def apply_hysteresis(scores, th_high, th_low):
    """Doppia soglia: scatta se superi la alta, si spegne se scendi sotto la bassa"""
    preds = np.zeros(len(scores), dtype=int)
    in_attack = False
    
    for i, s in enumerate(scores):
        if s >= th_high:
            preds[i] = 1
            in_attack = True
        elif s >= th_low and in_attack:
            preds[i] = 1
        else:
            preds[i] = 0
            in_attack = False
            
    # Riavvolgimento all'indietro per prendere l'inizio dell'attacco
    in_attack_rev = False
    for i in range(len(scores)-1, -1, -1):
        s = scores[i]
        if s >= th_high:
            preds[i] = 1
            in_attack_rev = True
        elif s >= th_low and in_attack_rev:
            preds[i] = 1
        else:
            in_attack_rev = False
            
    return preds

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
    y_test = df_test['True_Label'].values

    # Scaler
    features = ['Single_Score', 'Delta_Single_Min', 'Contex_Range', 'Time_Gradient']
    scaler = RobustScaler()
    X_val = scaler.fit_transform(df_val[features].values)
    X_test = scaler.transform(df_test[features].values)

    print("\nAddestramento OCSVM (TOP 1 Configuration)...")
    clf = OneClassSVM(nu=0.25, kernel='rbf', gamma=1.0)
    clf.fit(X_val)

    # Score invertiti (Alto = Anomalia)
    val_raw = -clf.decision_function(X_val)
    test_raw = -clf.decision_function(X_test)

    # Smoothing Temporale della Top 1 (EWMA=2)
    val_smooth = pd.Series(val_raw).ewm(span=2, adjust=False).mean().values
    test_smooth = pd.Series(test_raw).ewm(span=2, adjust=False).mean().values

    # =========================================================================
    # 1. RISULTATO BASE (Soglia Singola FPR 0.5%)
    # =========================================================================
    th_base = np.percentile(val_smooth, 100.0 - 0.5)
    preds_base = (test_smooth > th_base).astype(int)
    print_report(y_test, preds_base, "1. OCSVM BASELINE (FPR 0.5%)")

    # =========================================================================
    # 2. RISULTATO CON FILTRO MORFOLOGICO
    # =========================================================================
    # close_size=5 significa che se c'è un "buco" di 4 pacchetti normali 
    # in mezzo a un attacco, lo riempie considerandolo attacco.
    preds_morph = apply_morphological_filters(preds_base, close_size=5, open_size=2)
    print_report(y_test, preds_morph, "2. OCSVM + MORPHOLOGICAL FILTERING (Chiusura Buchi)")

    # =========================================================================
    # 3. RISULTATO CON ISTERESI (Doppia Soglia)
    # =========================================================================
    th_high = np.percentile(val_smooth, 100.0 - 0.5) # Soglia severa
    th_low = np.percentile(val_smooth, 100.0 - 2.0)  # Soglia rilassata
    preds_hyst = apply_hysteresis(test_smooth, th_high, th_low)
    print_report(y_test, preds_hyst, "3. OCSVM + HYSTERESIS THRESHOLDING (Doppia Soglia)")

if __name__ == "__main__":
    main()