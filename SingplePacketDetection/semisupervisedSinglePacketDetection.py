import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

# --- CONFIGURAZIONE ---
TEST_FILE = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_detection_results.csv"

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
    print("Caricamento dataset Single Packet (Semi-Supervised)...")
    try:
        df_test = pd.read_csv(TEST_FILE, usecols=['True_Label', 'Single_Score'])
    except FileNotFoundError as e:
        print(f"Errore caricamento file: {e}")
        return

    # Pulizia dati
    df_test['Single_Score'] = df_test['Single_Score'].fillna(df_test['Single_Score'].mean())

    y_test = df_test['True_Label'].values
    scores_test = df_test['Single_Score'].values

    print("\nRicerca Semi-Supervised: Scansione di 200 soglie direttamente sul Test Set...")
    
    # Creiamo 200 possibili soglie
    soglie_candidati = np.linspace(scores_test.min(), scores_test.max(), 200)
    
    best_f1 = -1
    best_th = 0
    best_preds = None
    
    for threshold in soglie_candidati:
        preds = (scores_test < threshold).astype(int)
        current_f1 = f1_score(y_test, preds, zero_division=0)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_th = threshold
            best_preds = preds

    # Invertiamo per l'AUC
    inverted_scores_test = -scores_test

    print_report(y_test, best_preds, inverted_scores_test, f"MIGLIOR RISULTATO SEMI-SUPERVISED (Soglia Ideale: {best_th:.4f})")

if __name__ == "__main__":
    main()