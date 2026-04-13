import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

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

    # Dizionari per tracciare i risultati migliori di ogni strategia
    best_avg = {'f1': -1, 'fpr': 0, 'th': 0, 'preds': None}
    best_min = {'f1': -1, 'fpr': 0, 'th': 0, 'preds': None}
    best_hybrid = {'f1': -1, 'fpr': 0, 'th_avg': 0, 'th_min': 0, 'preds': None}

    print(f"\nRicerca delle soglie basate sul Validation Set...")
    print(f"Filtro di persistenza temporale: {PERSISTENCE_WINDOW} step")
    
    for fpr in TARGET_FPRS:
        # Troviamo i percentili corretti sul validation (FPR% dal fondo)
        th_avg = np.percentile(val_sani_avg, fpr)
        th_min = np.percentile(val_sani_min, fpr)
        
        # Generiamo le raw predictions per le 3 strategie
        raw_preds_avg = (test_avg < th_avg).astype(int)
        raw_preds_min = (test_min < th_min).astype(int)
        raw_preds_hybrid = ((test_avg < th_avg) | (test_min < th_min)).astype(int)
        
        # Applichiamo la persistenza a tutte
        final_preds_avg = apply_persistence_filter(raw_preds_avg, window=PERSISTENCE_WINDOW)
        final_preds_min = apply_persistence_filter(raw_preds_min, window=PERSISTENCE_WINDOW)
        final_preds_hybrid = apply_persistence_filter(raw_preds_hybrid, window=PERSISTENCE_WINDOW)
        
        # Calcoliamo gli F1-Score
        f1_avg = f1_score(y_test, final_preds_avg, zero_division=0)
        f1_min = f1_score(y_test, final_preds_min, zero_division=0)
        f1_hybrid = f1_score(y_test, final_preds_hybrid, zero_division=0)
        
        # --- Aggiornamento dei record migliori ---
        if f1_avg > best_avg['f1']:
            best_avg.update({'f1': f1_avg, 'fpr': fpr, 'th': th_avg, 'preds': final_preds_avg})
            
        if f1_min > best_min['f1']:
            best_min.update({'f1': f1_min, 'fpr': fpr, 'th': th_min, 'preds': final_preds_min})
            
        if f1_hybrid > best_hybrid['f1']:
            best_hybrid.update({'f1': f1_hybrid, 'fpr': fpr, 'th_avg': th_avg, 'th_min': th_min, 'preds': final_preds_hybrid})

    # --- STAMPA DEI REPORT FINALI ---
    print_report(y_test, best_avg['preds'], test_avg, test_min, 
                 f"1. STRATEGIA 'SOLO MEDIA' (Ottimizzata per FPR: {best_avg['fpr']}%)")
                 
    print_report(y_test, best_min['preds'], test_avg, test_min, 
                 f"2. STRATEGIA 'SOLO MINIMO' (Ottimizzata per FPR: {best_min['fpr']}%)")
                 
    print_report(y_test, best_hybrid['preds'], test_avg, test_min, 
                 f"3. STRATEGIA 'IBRIDA [Avg OR Min]' (Ottimizzata per FPR: {best_hybrid['fpr']}%)")

    # --- CLASSIFICA ---
    print("\n" + "#"*70)
    print(" CLASSIFICA FINALE F1-SCORE (SEQUENCE DETECTION)")
    print("#"*70)
    
    classifica = [
        ("Solo Media", best_avg['f1']),
        ("Solo Minimo", best_min['f1']),
        ("Ibrida (Media OR Minimo)", best_hybrid['f1'])
    ]
    # Ordiniamo dal migliore al peggiore
    classifica.sort(key=lambda x: x[1], reverse=True)
    
    for i, (nome, punteggio) in enumerate(classifica, 1):
        if i == 1:
            print(f" 🏆 {nome:<25}: {punteggio:.4f}")
        else:
            print(f" {i}. {nome:<25}: {punteggio:.4f}")

if __name__ == "__main__":
    main()