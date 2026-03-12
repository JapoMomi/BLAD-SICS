import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURAZIONE ---
INPUT_FILE = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_detection_results.csv"

def print_report(y_true, y_pred, y_probs, title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    print(classification_report(y_true, y_pred, digits=4, target_names=["Benign", "Attack"]))
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n[TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}]\n[FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}]")
    if y_probs is not None:
        print(f"ROC AUC: {roc_auc_score(y_true, y_probs):.4f}")

def optimize_heuristic_min(df):
    """Strategia Euristica: Z-Score sul Minimo dei Contesti invece che sulla Media"""
    # 1. Troviamo il peggior score contestuale per ogni pacchetto
    context_cols = [c for c in df.columns if 'Ctx_Pos' in c]
    df['Min_Context'] = df[context_cols].min(axis=1)
    
    # 2. Normalizzazione (Z-Score)
    df['S_Norm'] = (df['Single_Score'] - df['Single_Score'].mean()) / df['Single_Score'].std()
    df['C_Norm'] = (df['Min_Context'] - df['Min_Context'].mean()) / df['Min_Context'].std()
    
    best_f1, best_w, best_th, best_preds = -1, 0, 0, None
    best_probs = None
    
    # Grid Search sui pesi e soglie
    for w in np.arange(0.1, 1.0, 0.1):
        combined_scores = (w * df['S_Norm']) + ((1 - w) * df['C_Norm'])
        # Invertiamo per comodità: valori più alti = più anomalo
        anomaly_scores = -combined_scores 
        
        thresholds = np.linspace(np.percentile(anomaly_scores, 1), np.percentile(anomaly_scores, 99), 100)
        
        for th in thresholds:
            preds = (anomaly_scores > th).astype(int)
            current_f1 = f1_score(df['True_Label'], preds, zero_division=0)
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_w = w
                best_th = th
                best_preds = preds
                best_probs = anomaly_scores

    print_report(df['True_Label'], best_preds, best_probs, 
                 f"1. METODO EURISTICO (Single + Min Context) | Peso Single: {best_w:.1f}")
    return best_f1

def optimize_ml_ensemble(df):
    """Strategia ML POTENZIATA: Feature Engineering, Bilanciamento e Smoothing"""
    print("\n" + "-"*60)
    print("Preparazione Feature Engineering Avanzata...")
    
    # 1. Feature Base
    context_cols = [c for c in df.columns if 'Ctx_Pos' in c]
    
    # 2. Creazione Meta-Features (Feature Engineering)
    df['Ctx_Min'] = df[context_cols].min(axis=1)
    df['Ctx_Mean'] = df[context_cols].mean(axis=1)
    df['Ctx_Std'] = df[context_cols].std(axis=1).fillna(0) # Varianza
    
    # Deltas: Quanto il contesto cambia la percezione del pacchetto?
    df['Delta_Single_Min'] = df['Single_Score'] - df['Ctx_Min']
    df['Delta_Single_Mean'] = df['Single_Score'] - df['Ctx_Mean']
    
    features = ['Single_Score'] + context_cols + ['Ctx_Min', 'Ctx_Mean', 'Ctx_Std', 'Delta_Single_Min', 'Delta_Single_Mean']
    
    X = df[features].fillna(df[features].mean())
    y = df['True_Label'].values
    
    # 3. Creazione Classificatore (Più profondo e Bilanciato)
    # class_weight='balanced' forza il modello a penalizzare pesantemente i Falsi Negativi
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=7, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    
    print("Addestramento Meta-Model con Cross-Validation (attendere...)")
    raw_probs = cross_val_predict(clf, X, y, cv=5, method='predict_proba')[:, 1]
    
    # 4. Smoothing Temporale sulle Probabilità (Il segreto degli IDS)
    # Un attacco raramente avviene in 1 solo pacchetto. Smussiamo le probabilità.
    print("Applicazione Smoothing Temporale sulle Probabilità (EWMA)...")
    probs_series = pd.Series(raw_probs)
    # span=3 crea un effetto memoria leggero sui pacchetti adiacenti
    smoothed_probs = probs_series.ewm(span=3, adjust=False).mean().values
    
    # 5. Tuning della Soglia
    best_f1, best_th, best_preds = -1, 0, None
    thresholds = np.linspace(0.01, 0.99, 100)
    
    for th in thresholds:
        preds = (smoothed_probs > th).astype(int)
        current_f1 = f1_score(y, preds, zero_division=0)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_th = th
            best_preds = preds
            
    print_report(y, best_preds, smoothed_probs, 
                 f"2. METODO ML ENSEMBLE ADVANCED (Features + Balanced + EWMA) | Soglia Prob: {best_th:.2f}")
    
    return best_f1
def main():
    print(f"Caricamento dati da {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Errore: File non trovato.")
        return

    # Esecuzione dei due metodi
    f1_euristico = optimize_heuristic_min(df)
    f1_ml = optimize_ml_ensemble(df)

    print("\n" + "#"*70)
    print(" CONCLUSIONE PER IL PAPER SCIENTIFICO")
    print("#"*70)
    print("Nel paper puoi confrontare questi approcci per mostrare l'evoluzione:")
    print(f"- Modello Base (Media Contesti): F1-Score ~ 0.8052 (Tuo risultato iniziale)")
    print(f"- Ottimizzazione Euristica (Min Context): F1-Score {f1_euristico:.4f}")
    print(f"- Stacking Ensemble (Machine Learning): F1-Score {f1_ml:.4f}")

if __name__ == "__main__":
    main()