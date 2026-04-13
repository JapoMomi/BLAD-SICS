import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURAZIONE ---
INPUT_FILE = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_detection_results.csv"

def print_report(y_true, y_pred, y_probs, title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    print(classification_report(y_true, y_pred, digits=4, target_names=["Benign", "Attack"]))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}")
    print(f"FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}")
    if y_probs is not None:
        print(f"ROC AUC: {roc_auc_score(y_true, y_probs):.4f}")

def optimize_heuristic_mean(df):
    context_cols = [c for c in df.columns if 'Ctx_Pos' in c]
    df['Mean_Context'] = df[context_cols].mean(axis=1)
    
    df['S_Norm'] = (df['Single_Score'] - df['Single_Score'].mean()) / df['Single_Score'].std()
    df['C_Norm'] = (df['Mean_Context'] - df['Mean_Context'].mean()) / df['Mean_Context'].std()
    
    best_f1, best_w, best_preds = -1, 0, None
    best_probs = None
    
    for w in np.arange(0.1, 1.0, 0.1):
        combined_scores = (w * df['S_Norm']) + ((1 - w) * df['C_Norm'])
        anomaly_scores = -combined_scores 
        thresholds = np.linspace(np.percentile(anomaly_scores, 1), np.percentile(anomaly_scores, 99), 100)
        
        for th in thresholds:
            preds = (anomaly_scores > th).astype(int)
            current_f1 = f1_score(df['True_Label'], preds, zero_division=0)
            if current_f1 > best_f1:
                best_f1, best_w, best_preds, best_probs = current_f1, w, preds, anomaly_scores

    print_report(df['True_Label'], best_preds, best_probs, 
                 f"1. EURISTICA BASE (Single + Mean Context) | Peso Single: {best_w:.1f}")
    return best_f1

def optimize_heuristic_min(df):
    context_cols = [c for c in df.columns if 'Ctx_Pos' in c]
    df['Min_Context'] = df[context_cols].min(axis=1)
    
    df['S_Norm'] = (df['Single_Score'] - df['Single_Score'].mean()) / df['Single_Score'].std()
    df['C_Norm'] = (df['Min_Context'] - df['Min_Context'].mean()) / df['Min_Context'].std()
    
    best_f1, best_w, best_preds = -1, 0, None
    best_probs = None
    
    for w in np.arange(0.1, 1.0, 0.1):
        combined_scores = (w * df['S_Norm']) + ((1 - w) * df['C_Norm'])
        anomaly_scores = -combined_scores 
        thresholds = np.linspace(np.percentile(anomaly_scores, 1), np.percentile(anomaly_scores, 99), 100)
        
        for th in thresholds:
            preds = (anomaly_scores > th).astype(int)
            current_f1 = f1_score(df['True_Label'], preds, zero_division=0)
            if current_f1 > best_f1:
                best_f1, best_w, best_preds, best_probs = current_f1, w, preds, anomaly_scores

    print_report(df['True_Label'], best_preds, best_probs, 
                 f"2. EURISTICA AGGRESSIVA (Single + Min Context) | Peso Single: {best_w:.1f}")
    return best_f1

def optimize_heuristic_delta(df):
    context_cols = [c for c in df.columns if 'Ctx_Pos' in c]
    df['Min_Context'] = df[context_cols].min(axis=1)
    df['Delta_Score'] = df['Single_Score'] - df['Min_Context']
    
    anomaly_scores = df['Delta_Score']
    best_f1, best_preds, best_probs = -1, None, anomaly_scores
    thresholds = np.linspace(np.percentile(anomaly_scores, 1), np.percentile(anomaly_scores, 99), 100)
    
    for th in thresholds:
        preds = (anomaly_scores > th).astype(int)
        current_f1 = f1_score(df['True_Label'], preds, zero_division=0)
        if current_f1 > best_f1:
            best_f1, best_preds = current_f1, preds

    print_report(df['True_Label'], best_preds, best_probs, "3. EURISTICA DELTA PURE (Single_Score - Min_Context)")
    return best_f1

def run_ml_ensembles(df):
    print("\n" + "-"*60)
    print("Preparazione Feature Engineering Avanzata...")
    
    context_cols = [c for c in df.columns if 'Ctx_Pos' in c]
    df['Ctx_Min'] = df[context_cols].min(axis=1)
    df['Ctx_Mean'] = df[context_cols].mean(axis=1)
    df['Ctx_Std'] = df[context_cols].std(axis=1).fillna(0)
    df['Delta_Single_Min'] = df['Single_Score'] - df['Ctx_Min']
    df['Delta_Single_Mean'] = df['Single_Score'] - df['Ctx_Mean']
    
    features = ['Single_Score'] + context_cols + ['Ctx_Min', 'Ctx_Mean', 'Ctx_Std', 'Delta_Single_Min', 'Delta_Single_Mean']
    
    X = df[features].fillna(df[features].mean())
    y = df['True_Label'].values
    
    # max_depth=7 aiuta molto a prevenire l'overfitting
    clf = RandomForestClassifier(n_estimators=200, max_depth=7, class_weight='balanced', random_state=42, n_jobs=-1)
    
    print("\nAddestramento Meta-Model con TimeSeriesSplit (Verifica Overfitting)...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    train_f1_scores, val_f1_scores = [], []
    raw_probs = np.zeros(len(y))
    val_indices_list = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        clf.fit(X_train, y_train)
        
        # Train Eval
        train_preds = clf.predict(X_train)
        train_f1 = f1_score(y_train, train_preds, zero_division=0)
        train_f1_scores.append(train_f1)
        
        # Validation Eval
        val_probs = clf.predict_proba(X_val)[:, 1]
        raw_probs[val_idx] = val_probs
        val_preds = (val_probs > 0.5).astype(int) # Usiamo 0.5 standard per il check
        val_f1 = f1_score(y_val, val_preds, zero_division=0)
        val_f1_scores.append(val_f1)
        
        val_indices_list.extend(val_idx)
        print(f"  Fold {fold+1} -> Train F1: {train_f1:.4f} | Validation F1: {val_f1:.4f}")

    mean_train_f1 = np.mean(train_f1_scores)
    mean_val_f1 = np.mean(val_f1_scores)
    
    print("\nRisultato Analisi Overfitting (Soglia grezza 0.5):")
    print(f"  -> MEDIA TRAIN F1: {mean_train_f1:.4f}")
    print(f"  -> MEDIA VAL F1:   {mean_val_f1:.4f}")
    if mean_train_f1 - mean_val_f1 > 0.10:
        print("  ⚠️ ATTENZIONE: Il modello mostra segni di overfitting (divario > 10%).")
    else:
        print("  ✅ OTTIMO: Nessun overfitting grave. Il modello generalizza bene!")

    # Per i report finali consideriamo solo i dati che sono stati nel set di validazione
    # (Il TimeSeriesSplit lascia i primi N pacchetti solo come train per avviare la serie)
    val_indices = np.array(val_indices_list)
    y_eval = y[val_indices]
    probs_eval = raw_probs[val_indices]
    
    # --- 4. ML ENSEMBLE BASE (NO SMOOTHING) ---
    best_f1_base, best_th_base, best_preds_base = -1, 0, None
    thresholds = np.linspace(0.01, 0.99, 100)
    
    for th in thresholds:
        preds = (probs_eval > th).astype(int)
        current_f1 = f1_score(y_eval, preds, zero_division=0)
        if current_f1 > best_f1_base:
            best_f1_base, best_th_base, best_preds_base = current_f1, th, preds
            
    print_report(y_eval, best_preds_base, probs_eval, 
                 f"4. ML ENSEMBLE BASE (Out-Of-Fold) | Soglia Prob: {best_th_base:.2f}")

    # --- 5. ML ENSEMBLE ADVANCED (CON SMOOTHING EWMA) ---
    print("\nApplicazione Smoothing Temporale (EWMA) in modo sicuro sui fold...")
    probs_series = pd.Series(probs_eval)
    smoothed_probs = probs_series.ewm(span=2, adjust=False).mean().values
    
    best_f1_adv, best_th_adv, best_preds_adv = -1, 0, None
    
    for th in thresholds:
        preds = (smoothed_probs > th).astype(int)
        current_f1 = f1_score(y_eval, preds, zero_division=0)
        if current_f1 > best_f1_adv:
            best_f1_adv, best_th_adv, best_preds_adv = current_f1, th, preds
            
    print_report(y_eval, best_preds_adv, smoothed_probs, 
                 f"5. ML ENSEMBLE ADVANCED (Out-Of-Fold + EWMA) | Soglia Prob: {best_th_adv:.2f}")
    
    return best_f1_base, best_f1_adv

def main():
    print(f"Caricamento dati da {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Errore: File non trovato.")
        return

    f1_mean = optimize_heuristic_mean(df.copy())
    f1_min = optimize_heuristic_min(df.copy())
    f1_delta = optimize_heuristic_delta(df.copy())
    f1_ml_base, f1_ml_adv = run_ml_ensembles(df.copy())

    print("\n" + "#"*80)
    print(" RIASSUNTO RISULTATI PER IL PAPER SCIENTIFICO")
    print("#"*80)
    print(f"1. Euristica Base (Single + Mean Context):   F1-Score {f1_mean:.4f}")
    print(f"2. Euristica Aggressiva (Single + Min Ctx):  F1-Score {f1_min:.4f}")
    print(f"3. Euristica Delta Pura (Discrepanza):       F1-Score {f1_delta:.4f}")
    print(f"4. ML Ensemble Base (Time Series CV):        F1-Score {f1_ml_base:.4f}")
    print(f"5. ML Ensemble Advanced (TS-CV + Smoothing): F1-Score {f1_ml_adv:.4f}")

if __name__ == "__main__":
    main()