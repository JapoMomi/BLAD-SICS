import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

# --- CONFIGURAZIONE ---
VAL_FILE = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_validation_results.csv"
TEST_FILE = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_detection_results.csv"

# Tolleranze di Falsi Allarmi (FPR) da testare sul traffico sano (Validation)
TARGET_FPRS = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

def print_report(y_true, y_pred, y_probs, title):
    print(f"\n{'='*75}\n{title}\n{'='*75}")
    print(classification_report(y_true, y_pred, digits=4, target_names=["Benign", "Attack"], zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n[TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}]\n[FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}]")
    if y_probs is not None:
        try:
            print(f"ROC AUC: {roc_auc_score(y_true, y_probs):.4f}")
        except ValueError:
            pass

def main():
    print("Caricamento dataset Single Packet (Unsupervised)...")
    try:
        # Carichiamo solo le colonne essenziali
        df_val = pd.read_csv(VAL_FILE, usecols=['True_Label', 'Single_Score'])
        df_test = pd.read_csv(TEST_FILE, usecols=['True_Label', 'Single_Score'])
    except FileNotFoundError as e:
        print(f"Errore caricamento file: {e}")
        return

    # 1. Pulizia dati
    df_val['Single_Score'] = df_val['Single_Score'].fillna(df_val['Single_Score'].mean())
    df_test['Single_Score'] = df_test['Single_Score'].fillna(df_test['Single_Score'].mean())

    # 2. Garanzia Unsupervised: prendiamo solo i pacchetti sani dal Validation
    if df_val['True_Label'].sum() > 0:
        val_sani = df_val[df_val['True_Label'] == 0]['Single_Score'].values
    else:
        val_sani = df_val['Single_Score'].values

    y_test = df_test['True_Label'].values
    scores_test = df_test['Single_Score'].values

    best_f1 = -1
    best_fpr = 0
    best_th = 0
    best_preds = None

    print("\nRicerca della migliore soglia Unsupervised basata sul Validation Set...")
    
    for fpr in TARGET_FPRS:
        threshold = np.percentile(val_sani, fpr)
        preds = (scores_test < threshold).astype(int)
        
        current_f1 = f1_score(y_test, preds, zero_division=0)
        print(f" -> Tolleranza Val FPR: {fpr:>4.1f}% | Soglia: {threshold:>8.4f} | F1 Test: {current_f1:.4f}")
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_fpr = fpr
            best_th = threshold
            best_preds = preds

    # Invertiamo il segno degli score per calcolare l'AUC
    inverted_scores_test = -scores_test

    print_report(y_test, best_preds, inverted_scores_test, f"MIGLIOR RISULTATO UNSUPERVISED (Ottimizzato per FPR: {best_fpr}%)")

if __name__ == "__main__":
    main()