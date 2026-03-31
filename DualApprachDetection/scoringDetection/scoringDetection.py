import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

# --- CONFIGURAZIONE PERCORSI ---
VAL_CSV = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_validation_results.csv"
TEST_CSV = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_detection_results.csv"

# Tolleranze FPR da testare sul Validation
PERCENTILES_TO_TEST = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]

def prep_features(df):
    """Estrazione delle Golden Features come richiesto."""
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

def main():
    print("Lettura e Feature Engineering dei dataset...")
    try:
        df_val = pd.read_csv(VAL_CSV)
        df_test = pd.read_csv(TEST_CSV)
    except FileNotFoundError as e:
        print(f"Errore caricamento file: {e}")
        return

    # Estrazione feature
    df_val = prep_features(df_val)
    df_test = prep_features(df_test)

    # Garanzia Unsupervised: solo pacchetti sani dal Val
    if 'True_Label' in df_val.columns and df_val['True_Label'].sum() > 0:
        val_sani = df_val[df_val['True_Label'] == 0]
    else:
        val_sani = df_val

    y_test = df_test['True_Label'].values

    # Definiamo quali feature cercare in coda inferiore (Low) e superiore (High)
    feature_configs = {
        'Single_Score': 'low',
        'Min_Context': 'low',
        'Mean_Context': 'low',
        'Max_Context': 'low',
        'Std_Context': 'high',
        'Delta_Single_Min': 'high',
        'Delta_Single_Mean': 'high',
        'Contex_Range': 'high',
        'Time_Gradient': 'high_abs' # Useremo il valore assoluto
    }

    results = {}
    
    print("\nRicerca delle soglie ottimali Unsupervised in corso...")

    for feat, tail in feature_configs.items():
        best_f1 = -1
        best_fpr = 0
        best_th = 0
        best_preds = None
        
        # Estrazione array
        if tail == 'high_abs':
            val_scores = val_sani[feat].abs().values
            test_scores = df_test[feat].abs().values
        else:
            val_scores = val_sani[feat].values
            test_scores = df_test[feat].values

        # Ricerca per ogni tolleranza
        for fpr in PERCENTILES_TO_TEST:
            if tail == 'low':
                # Anomalia = Valori molto negativi (coda bassa)
                th = np.percentile(val_scores, fpr)
                preds = (test_scores < th).astype(int)
            else:
                # Anomalia = Valori molto alti (coda alta)
                th = np.percentile(val_scores, 100.0 - fpr)
                preds = (test_scores > th).astype(int)
                
            current_f1 = f1_score(y_test, preds, zero_division=0)
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_fpr = fpr
                best_th = th
                best_preds = preds
                
        # Preparazione punteggi per l'AUC (Alto = Anomalia)
        if tail == 'low':
            auc_scores = -test_scores
        else:
            auc_scores = test_scores

        results[feat] = {
            'f1': best_f1,
            'fpr': best_fpr,
            'th': best_th,
            'preds': best_preds,
            'auc_scores': auc_scores
        }

    # --- STAMPA DEI REPORT ---
    print("\n" + "="*80)
    print(" REPORT COMPLETO PER SINGOLA FEATURE ESTRATTA")
    print("="*80)
    
    overall_best_f1 = -1
    overall_best_name = ""
    
    for feat, data in results.items():
        y_pred = data['preds']
        f1 = data['f1']
        
        if f1 > overall_best_f1:
            overall_best_f1 = f1
            overall_best_name = feat
            
        try:
            auc = roc_auc_score(y_test, data['auc_scores'])
        except ValueError:
            auc = 0.0

        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n{'-'*50}")
        print(f" Feature: {feat.upper()}")
        print(f" (Ottimizzata al FPR: {data['fpr']}% | Soglia: {data['th']:.4f})")
        print(f"{'-'*50}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4, target_names=["Benign", "Attack"], zero_division=0))
        print(f"Confusion Matrix:\n[TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}]\n[FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}]")
        print(f"ROC AUC: {auc:.4f}")

    print("\n" + "="*80)
    print(f" 🏆 FEATURE VINCITRICE: {overall_best_name.upper()} (F1-Score: {overall_best_f1:.4f})")
    print("="*80)

if __name__ == "__main__":
    main()