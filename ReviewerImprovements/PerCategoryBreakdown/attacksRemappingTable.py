import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# --- CONFIGURAZIONE PERCORSI ---
FULL_DATASET_FILE = os.path.join(SCRIPT_DIR, "../../Dataset/IanRawDataset.txt")
VAL_CSV = os.path.join(SCRIPT_DIR, "../../DualModelDetection/dual_model_validation_results.csv")
TEST_CSV = os.path.join(SCRIPT_DIR, "../../DualModelDetection/dual_model_detection_results.csv")

COL_IDX_LABEL1 = 2  
COL_IDX_TIME = 5

def load_and_slice_raw_dataset(filepath):
    print("Caricamento dataset grezzo originale...")
    df = pd.read_csv(filepath, header=None, dtype=str)
    df['ts_float'] = df[COL_IDX_TIME].astype(float)
    df = df.sort_values('ts_float')
    
    # Stesso taglio 85% usato per creare il test set
    idx_val_end = int(len(df) * 0.85)
    df_test = df.iloc[idx_val_end:].reset_index(drop=True)
    return df_test

def map_attack_category(attack_id_str):
    """Mappa l'ID dell'attacco alla sua categoria in base alla Tabella 1 del Paper."""
    try:
        aid = int(attack_id_str)
    except:
        return "Benign"
        
    if aid == 0: return "Benign"
    if 1 <= aid <= 12: return "MPCI"
    if 13 <= aid <= 17: return "MSCI"
    if aid == 18: return "DoS"
    if aid in [19, 21, 22]: return "MFCI"
    if aid in [20, 23, 24]: return "Recon"
    if aid in [25, 26, 27, 28, 33, 34, 35]: return "CMRI"
    if 29 <= aid <= 32: return "NMRI"
    return "Unknown"

def prep_features(df):
    """Stessa esatta estrazione features usata nel Dual Model."""
    context_cols = [c for c in df.columns if 'Ctx_Pos' in c]
    df[context_cols] = df[context_cols].apply(lambda row: row.fillna(row.mean()), axis=1)
    df['Single_Score'] = df['Single_Score'].fillna(df['Single_Score'].mean())
    
    if 'Min_Single_Score' in df.columns:
        df['Min_Single_Score'] = df['Min_Single_Score'].fillna(df['Min_Single_Score'].mean())
        
    df['Min_Context'] = df[context_cols].min(axis=1)
    df['Mean_Context'] = df[context_cols].mean(axis=1)
    df['Max_Context'] = df[context_cols].max(axis=1)
    
    df['Delta_Single_Min'] = df['Single_Score'] - df['Min_Context']
    df['Contex_Range'] = df['Max_Context'] - df['Min_Context']

    if 'Ctx_Pos0' in df.columns and 'Ctx_Pos4' in df.columns:
        df['Time_Gradient'] = df['Ctx_Pos4'] - df['Ctx_Pos0']
    else:
        df['Time_Gradient'] = 0.0
        
    return df

def main():
    # 1. Recupero le categorie degli attacchi originali
    df_test_raw = load_and_slice_raw_dataset(FULL_DATASET_FILE)
    df_test_raw['Category'] = df_test_raw[COL_IDX_LABEL1].apply(map_attack_category)
    
    print("Caricamento file Validation e Test CSV...")
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)
    
    if len(df_test_raw) != len(df_test):
        print(f"ERRORE: Le lunghezze non combaciano! (Raw: {len(df_test_raw)}, Test CSV: {len(df_test)})")
        return
        
    df_test['Category'] = df_test_raw['Category']
    
    # Prep features
    df_val = prep_features(df_val)
    df_test = prep_features(df_test)
    
    # 2. Generazione Predizioni Single Packet (Min) a FPR 1%
    print("Calcolo predizioni Single Packet (Minimo)...")
    if 'True_Label' in df_val.columns and df_val['True_Label'].sum() > 0:
        val_sani_single = df_val[df_val['True_Label'] == 0]['Min_Single_Score'].values
    else:
        val_sani_single = df_val['Min_Single_Score'].values
        
    th_single = np.percentile(val_sani_single, 1.0)
    df_test['Pred_Single'] = (df_test['Min_Single_Score'] < th_single).astype(int)
    
    # 3. Generazione Predizioni Dual-Model OCSVM
    print("Calcolo predizioni Dual-Model (OCSVM)...")
    features = ['Single_Score', 'Min_Single_Score', 'Delta_Single_Min', 'Contex_Range', 'Time_Gradient']
    
    if 'True_Label' in df_val.columns and df_val['True_Label'].sum() > 0:
        df_val_sano = df_val[df_val['True_Label'] == 0].copy()
    else:
        df_val_sano = df_val.copy()
        
    # Ripristinato il RobustScaler originale
    scaler = RobustScaler()
    X_val = scaler.fit_transform(df_val_sano[features].values)
    X_test = scaler.transform(df_test[features].values)
    
    clf = OneClassSVM(nu=0.05, gamma=0.05, kernel='rbf')
    clf.fit(X_val)
    
    val_raw_ocsvm = -clf.decision_function(X_val)
    val_smooth_ocsvm = pd.Series(val_raw_ocsvm).ewm(span=3, adjust=False).mean().values
    
    test_raw_ocsvm = -clf.decision_function(X_test)
    test_smooth_ocsvm = pd.Series(test_raw_ocsvm).ewm(span=3, adjust=False).mean().values
    
    th_ocsvm = np.percentile(val_smooth_ocsvm, 100.0 - 1.0)
    df_test['Pred_Dual'] = (test_smooth_ocsvm > th_ocsvm).astype(int)

    # 4. Costruzione della Tabella (Per-Attack Category Breakdown)
    print("\n" + "="*80)
    print(f"{'Category':<15} | {'Tot Attacks':<12} | {'Single Packet Recall':<22} | {'Dual-Model Recall':<20}")
    print("="*80)
    
    attacks_only = df_test[df_test['True_Label'] == 1]
    categories = ['MPCI', 'MSCI', 'CMRI', 'NMRI', 'MFCI', 'Recon', 'DoS']
    
    # Variabili per i totali
    total_attacks_count = 0
    total_single_det = 0
    total_dual_det = 0
    
    for cat in categories:
        cat_df = attacks_only[attacks_only['Category'] == cat]
        tot = len(cat_df)
        if tot == 0:
            continue
            
        single_det = cat_df['Pred_Single'].sum()
        dual_det = cat_df['Pred_Dual'].sum()
        
        # Aggiornamento totali
        total_attacks_count += tot
        total_single_det += single_det
        total_dual_det += dual_det
        
        single_recall = single_det / tot
        dual_recall = dual_det / tot
        
        print(f"{cat:<15} | {tot:<12} | {single_det:>5} ({single_recall*100:>5.1f}%)         | {dual_det:>5} ({dual_recall*100:>5.1f}%)")
    
    # Stampa della riga dei Totali
    print("-" * 80)
    if total_attacks_count > 0:
        overall_single_recall = total_single_det / total_attacks_count
        overall_dual_recall = total_dual_det / total_attacks_count
        print(f"{'TOTAL':<15} | {total_attacks_count:<12} | {total_single_det:>5} ({overall_single_recall*100:>5.1f}%)         | {total_dual_det:>5} ({overall_dual_recall*100:>5.1f}%)")
    
    print("="*80)

if __name__ == "__main__":
    main()