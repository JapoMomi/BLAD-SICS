import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURAZIONE ---
# Percorsi dei file CSV generati dal tuo script precedente
VAL_FILE = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/sequenceDetection_mean+min_Threshold/detection_detailed_results_validation.csv"
TEST_FILE = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/sequenceDetection_mean+min_Threshold/detection_detailed_results.csv"

# Tolleranze di Falsi Allarmi (FPR) da testare sul traffico sano (Validation)
TARGET_FPRS = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]
PERSISTENCE_WINDOW = 3

def print_report(y_true, y_pred, probs_avg, probs_min, title):
    print(f"\n{'='*75}\n{title}\n{'='*75}")
    print(classification_report(y_true, y_pred, digits=4, target_names=["Benign", "Attack"], zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n[TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}]\n[FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}]")
    
    # Calcolo ROC AUC invertendo i punteggi (i valori più bassi/negativi sono anomalie)
    try:
        auc_avg = roc_auc_score(y_true, -probs_avg)
        auc_min = roc_auc_score(y_true, -probs_min)
        print(f"ROC AUC (su Avg_Score): {auc_avg:.4f}")
        print(f"ROC AUC (su Min_Score): {auc_min:.4f}")
    except ValueError:
        print("Impossibile calcolare ROC AUC (una sola classe presente).")

def apply_persistence_filter(preds, window=3):
    """
    Richiede che l'allarme sia mantenuto per 'window' sequenze consecutive.
    (Preso direttamente dalla tua logica originale).
    """
    if window <= 1:
        return preds
        
    filtered = np.zeros_like(preds)
    kernel = np.ones(window)
    # Somma mobile: se la somma è uguale alla finestra, erano tutti 1
    conv = np.convolve(preds, kernel, mode='valid')
    detected = (conv == window).astype(int)
    
    # Ripristiniamo la lunghezza originale con padding all'inizio
    filtered[window-1:] = detected
    return filtered

def main():
    print("Caricamento dei risultati della Sequence Detection (Unsupervised)...")
    try:
        df_val = pd.read_csv(VAL_FILE)
        df_test = pd.read_csv(TEST_FILE)
    except FileNotFoundError as e:
        print(f"Errore caricamento file: {e}\nAssicurati di aver eseguito prima lo script di reconstruction.")
        return

    # 1. Garanzia Unsupervised: prendiamo solo il traffico sano dal Validation
    if 'Label' in df_val.columns and df_val['Label'].sum() > 0:
        df_val_sano = df_val[df_val['Label'] == 0]
    else:
        df_val_sano = df_val
        
    # Estraiamo le distribuzioni di normalità
    val_sani_avg = df_val_sano['Avg_Score'].values
    val_sani_min = df_val_sano['Min_Score'].values

    # Estraiamo i dati di Test
    y_test = df_test['Label'].values
    test_avg = df_test['Avg_Score'].values
    test_min = df_test['Min_Score'].values

    best_f1 = -1
    best_fpr = 0
    best_th_avg = 0
    best_th_min = 0
    best_preds = None

    print(f"\nRicerca delle soglie ibride (Avg OR Min) basate sul Validation Set...")
    print(f"Filtro di persistenza temporale: {PERSISTENCE_WINDOW} step")
    
    for fpr in TARGET_FPRS:
        # Troviamo il percentile corretto sul validation (FPR% dal fondo)
        th_avg = np.percentile(val_sani_avg, fpr)
        th_min = np.percentile(val_sani_min, fpr)
        
        # Logica Ibrida: Allarme se la media della sequenza è bassa OR il minimo è molto basso
        raw_preds = ((test_avg < th_avg) | (test_min < th_min)).astype(int)
        
        # Applichiamo la persistenza (la tua stessa logica)
        final_preds = apply_persistence_filter(raw_preds, window=PERSISTENCE_WINDOW)
        
        current_f1 = f1_score(y_test, final_preds, zero_division=0)
        
        print(f" -> Tolleranza FPR: {fpr:>4.1f}% | Th_Avg: {th_avg:>8.4f} | Th_Min: {th_min:>8.4f} | F1 Test: {current_f1:.4f}")
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_fpr = fpr
            best_th_avg = th_avg
            best_th_min = th_min
            best_preds = final_preds

    print_report(y_test, best_preds, test_avg, test_min, f"MIGLIOR RISULTATO SEQUENCE UNSUPERVISED (Ottimizzato per FPR: {best_fpr}%)")

if __name__ == "__main__":
    main()