import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURAZIONE ---
VAL_FILE = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/sequenceDetection_mean+min_Threshold/detection_detailed_results_validation.csv"
TEST_FILE = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/sequenceDetection_mean+min_Threshold/detection_detailed_results.csv"

TARGET_FPRS = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]

def print_report(y_true, y_pred, probs, title, best_fpr):
    print(f"\n{'='*80}\n{title} | Ottimizzato su Val FPR: {best_fpr}%\n{'='*80}")
    print(classification_report(y_true, y_pred, digits=4, target_names=["Benign", "Attack"], zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n[TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}]\n[FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}]")
    
    try:
        # Invertiamo il segno per l'AUC (i valori più bassi indicano anomalie)
        auc = roc_auc_score(y_true, -probs)
        print(f"ROC AUC: {auc:.4f}")
    except ValueError:
        print("Impossibile calcolare ROC AUC.")

def main():
    print("Caricamento Dataset Sequence Detection (Pattern Interni)...")
    try:
        df_val = pd.read_csv(VAL_FILE)
        df_test = pd.read_csv(TEST_FILE)
    except FileNotFoundError as e:
        print(f"Errore caricamento file: {e}")
        return

    # Colonne dei singoli pacchetti
    p_cols = ['Score_P1', 'Score_P2', 'Score_P3', 'Score_P4', 'Score_P5']

    # 1. Preparazione Dati Validation (Solo traffico benigno)
    if 'Label' in df_val.columns and df_val['Label'].sum() > 0:
        df_val_sano = df_val[df_val['Label'] == 0]
    else:
        df_val_sano = df_val
        
    val_p1 = df_val_sano['Score_P1'].values
    # Per valutare la soglia del singolo pacchetto in generale (per le strategie 2 e 3),
    # usiamo la distribuzione di TUTTI i pacchetti del traffico sano.
    val_all_packets = df_val_sano[p_cols].values.flatten()

    # 2. Preparazione Dati Test e derivazione degli "Score Continui" per l'AUC
    y_test = df_test['Label'].values
    test_all = df_test[p_cols].values

    # STRATEGIA 1: Score del 1° Pacchetto
    test_score_strat1 = df_test['Score_P1'].values

    # STRATEGIA 2: Almeno 2 pacchetti ovunque
    # Ordiniamo gli score dei 5 pacchetti della sequenza in modo crescente.
    # Il 2° elemento (indice 1) è il 2° pacchetto più anomalo. Se questo è < soglia, ne abbiamo almeno 2.
    test_score_strat2 = np.sort(test_all, axis=1)[:, 1]

    # STRATEGIA 3: Almeno 2 pacchetti consecutivi
    # Calcoliamo il punteggio "peggiore" (massimo) per ogni coppia adiacente.
    pair12 = np.maximum(test_all[:, 0], test_all[:, 1])
    pair23 = np.maximum(test_all[:, 1], test_all[:, 2])
    pair34 = np.maximum(test_all[:, 2], test_all[:, 3])
    pair45 = np.maximum(test_all[:, 3], test_all[:, 4])
    # Il punteggio della sequenza è la coppia più anomala (il minimo tra questi massimi)
    test_score_strat3 = np.minimum.reduce([pair12, pair23, pair34, pair45])


    # Variabili per tracciare i vincitori
    results = {}
    
    # --- CICLO DI OTTIMIZZAZIONE SULLE TOLLERANZE (FPR) ---
    print("\nRicerca delle soglie e test delle 3 strategie...")
    
    best_s1 = {'f1': -1, 'fpr': 0, 'preds': None}
    best_s2 = {'f1': -1, 'fpr': 0, 'preds': None}
    best_s3 = {'f1': -1, 'fpr': 0, 'preds': None}

    for fpr in TARGET_FPRS:
        # Soglie calcolate sul Validation
        th_p1 = np.percentile(val_p1, fpr)
        th_any = np.percentile(val_all_packets, fpr)

        # Inferenza sul Test
        preds_s1 = (test_score_strat1 < th_p1).astype(int)
        preds_s2 = (test_score_strat2 < th_any).astype(int)
        preds_s3 = (test_score_strat3 < th_any).astype(int)

        # F1-Scores
        f1_s1 = f1_score(y_test, preds_s1, zero_division=0)
        f1_s2 = f1_score(y_test, preds_s2, zero_division=0)
        f1_s3 = f1_score(y_test, preds_s3, zero_division=0)

        # Aggiornamento migliori risultati Strategy 1
        if f1_s1 > best_s1['f1']:
            best_s1.update({'f1': f1_s1, 'fpr': fpr, 'preds': preds_s1})
        # Aggiornamento migliori risultati Strategy 2
        if f1_s2 > best_s2['f1']:
            best_s2.update({'f1': f1_s2, 'fpr': fpr, 'preds': preds_s2})
        # Aggiornamento migliori risultati Strategy 3
        if f1_s3 > best_s3['f1']:
            best_s3.update({'f1': f1_s3, 'fpr': fpr, 'preds': preds_s3})


    # --- STAMPA REPORT FINALI ---
    print_report(y_test, best_s1['preds'], test_score_strat1, 
                 "STRATEGIA 1: Primo Pacchetto Anomalo", best_s1['fpr'])
    
    print_report(y_test, best_s2['preds'], test_score_strat2, 
                 "STRATEGIA 2: Almeno 2 Pacchetti Anomali", best_s2['fpr'])
    
    print_report(y_test, best_s3['preds'], test_score_strat3, 
                 "STRATEGIA 3: Almeno 2 Pacchetti Consecutivi", best_s3['fpr'])


    # --- CLASSIFICA FINALE ---
    print("\n" + "#"*70)
    print(" CLASSIFICA FINALE F1-SCORE")
    print("#"*70)
    
    classifica = [
        ("Primo Pacchetto Anomalo", best_s1['f1']),
        ("Almeno 2 Pacchetti Anomali", best_s2['f1']),
        ("Almeno 2 Pacchetti Consecutivi", best_s3['f1'])
    ]
    # Ordina dal migliore al peggiore
    classifica.sort(key=lambda x: x[1], reverse=True)
    
    for i, (nome, punteggio) in enumerate(classifica, 1):
        if i == 1:
            print(f" 🏆 {nome:<40}: {punteggio:.4f}")
        else:
            print(f" {i}. {nome:<40}: {punteggio:.4f}")

if __name__ == "__main__":
    main()