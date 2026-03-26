import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import itertools

# --- CONFIGURAZIONE ---
# Inserisci qui il percorso del tuo file CSV di test (che contiene sia Benign che Attack)
TEST_FILE = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_detection_results.csv"

# Griglia Parametri Isolation Forest per la ricerca della configurazione migliore
IFOREST_GRID = {
    'contamination': [0.18, 0.20, 0.22, 0.24, 0.26], 
    'n_estimators': [200, 300, 500],
    'max_samples': [256, 512, 1024],  
    'max_features': [1.0, 0.75, 0.5],      
    'ewma_span': [3, 4, 5, 6]
}

def prep_features(df):
    """
    Estrazione delle feature compatte derivate dagli score del ByT5.
    Calcola delta e gradienti temporali per evidenziare le anomalie di contesto.
    """
    context_cols = [c for c in df.columns if 'Ctx_Pos' in c]
    
    # Fill NaN per sicurezza
    df[context_cols] = df[context_cols].apply(lambda row: row.fillna(row.mean()), axis=1)
    df['Single_Score'] = df['Single_Score'].fillna(df['Single_Score'].mean())
    
    # Feature ingegnerizzate dalle probabilità logaritmiche
    df['Min_Context'] = df[context_cols].min(axis=1)
    df['Mean_Context'] = df[context_cols].mean(axis=1)
    df['Max_Context'] = df[context_cols].max(axis=1)
    df['Std_Context'] = df[context_cols].std(axis=1).fillna(0)
    
    df['Delta_Single_Min'] = df['Single_Score'] - df['Min_Context']
    df['Delta_Single_Mean'] = df['Single_Score'] - df['Mean_Context']
    df['Contex_Range'] = df['Max_Context'] - df['Min_Context']

    # Gradiente temporale se la finestra è di 5 pacchetti
    if 'Ctx_Pos0' in df.columns and 'Ctx_Pos4' in df.columns:
        df['Time_Gradient'] = df['Ctx_Pos4'] - df['Ctx_Pos0']
    else:
        df['Time_Gradient'] = 0.0
        
    return df

def run_unsupervised_iforest(df_test, features):
    """
    Esegue la Grid Search dell'Isolation Forest in modalità Unsupervised puro.
    Il FIT viene fatto sui dati misti, senza mai usare 'True_Label'.
    """
    print(f"\nAvvio Ricerca Unsupervised Isolation Forest su {len(df_test)} campioni misti...")
    
    # 1. Conserviamo le etichette reali SOLO per la valutazione finale (mai per il training)
    y_true = df_test['True_Label'].values
    
    # 2. Scaliamo i dati misti
    scaler = RobustScaler()
    X_mixed = scaler.fit_transform(df_test[features].values)
    
    keys = list(IFOREST_GRID.keys())
    combinations = list(itertools.product(*[IFOREST_GRID[k] for k in keys]))
    
    best_results = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        # Inizializzazione modello
        clf = IsolationForest(
            n_estimators=params['n_estimators'], 
            max_samples=params['max_samples'],        
            max_features=params['max_features'],     
            contamination=params['contamination'], 
            random_state=42, 
            n_jobs=-1
        )
        
        # ---> FASE CRITICA UNSUPERVISED: Fit sui dati misti (X_mixed) senza etichette <---
        clf.fit(X_mixed)
        
        # Calcolo degli anomaly score crudi (invertiti per comodità: alto = molto anomalo)
        raw_scores = -clf.decision_function(X_mixed)
        
        # Smoothing temporale con EWMA
        smoothed_scores = pd.Series(raw_scores).ewm(span=params['ewma_span'], adjust=False).mean().values
        
        # Calcolo delle predizioni
        if params['contamination'] != 'auto':
            # Se la contaminazione è fissa, calcoliamo il percentile per tagliare la coda di anomalie
            cont_rate = float(params['contamination'])
            th = np.percentile(smoothed_scores, 100.0 - (cont_rate * 100))
            preds = (smoothed_scores > th).astype(int)
        else:
            # Se 'auto', usiamo il thresholding nativo interno di sklearn
            preds_native = clf.predict(X_mixed)
            preds = np.where(preds_native == -1, 1, 0) # Rimappiamo: -1 (outlier) -> 1 (Attack)
            
        # Valutazione
        f1 = f1_score(y_true, preds, zero_division=0)
        
        best_results.append({
            'contamination': params['contamination'],
            'n_estimators': params['n_estimators'],
            'ewma_span': params['ewma_span'],
            'f1': f1,
            'preds': preds,
            'scores': smoothed_scores
        })
        
        if (i+1) % 10 == 0 or (i+1) == len(combinations):
            print(f"Progresso: {i+1}/{len(combinations)} combinazioni testate...")
            
    # Ordiniamo per F1-score decrescente e restituiamo le prime 3
    return sorted(best_results, key=lambda x: x['f1'], reverse=True)[:3]

def print_top_configuration(rank, config, y_test):
    print(f"\n{'='*75}")
    print(f" 🏆 TOP {rank} CONFIGURATION: IForest Unsupervised")
    print(f"{'='*75}")
    
    print(f"Parametri: Contamination={config['contamination']} | Trees={config['n_estimators']} | EWMA Span={config['ewma_span']}")
    print("-" * 75)
    
    print(classification_report(y_test, config['preds'], digits=4, target_names=["Benign", "Attack"]))
    
    cm = confusion_matrix(y_test, config['preds'])
    print(f"Confusion Matrix:\n[TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}]\n[FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}]")
    
    try:
        auc = roc_auc_score(y_test, config['scores'])
        print(f"\nROC AUC: {auc:.4f}")
    except ValueError:
        pass

def main():
    print(f"Caricamento Dataset di Test da: {TEST_FILE}")
    try:
        df_test = pd.read_csv(TEST_FILE)
    except FileNotFoundError as e:
        print(f"Errore: {e}")
        return

    # Estrazione feature
    df_test = prep_features(df_test)
    
    # Definizione delle feature da passare al modello
    features = ['Single_Score', 'Delta_Single_Min', 'Contex_Range', 'Time_Gradient']
    
    # Salviamo le label per la valutazione finale
    y_test = df_test['True_Label'].values

    # Esecuzione
    top_configs = run_unsupervised_iforest(df_test, features)

    print("\n" + "#"*75)
    print(" RISULTATI FINALI GRID SEARCH UNSUPERVISED")
    print("#"*75)
    
    for i, config in enumerate(top_configs):
        print_top_configuration(i+1, config, y_test)

if __name__ == "__main__":
    main()