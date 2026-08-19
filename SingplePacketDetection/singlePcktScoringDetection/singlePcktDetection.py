import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve # <-- NUOVI IMPORT
import os 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# --- CONFIGURAZIONE ---
VAL_FILE = os.path.join(SCRIPT_DIR, "../../DualModelDetection/dual_model_validation_results.csv")
TEST_FILE = os.path.join(SCRIPT_DIR, "../../DualModelDetection/dual_model_detection_results.csv")

# Tolleranze di Falsi Allarmi (FPR) da testare sul traffico sano (Validation)
TARGET_FPRS = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0,]

# --- PARAMETRI PER LE METRICHE AVANZATE (REVIEWER) ---
TEST_SET_HOURS = 76.58     # <-- AGGIORNA QUESTO VALORE CON LE ORE REALI DEL TEST SET
TARGET_FPR_MATCH = 0.0112 # <-- FPR 1.12% da forzare/verificare (Tabella 3)

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

def advanced_evaluation_for_reviewers(y_true, y_probs, model_name):
    """Calcola PR-AUC, Matched FPR performance, Alarms/Hour e salva la curva PR."""
    print(f"\n{'-'*75}")
    print(f" ADVANCED METRICS FOR REVIEWER (CONCERNS 1 & 3): {model_name}")
    print(f"{'-'*75}")

    # 1. PR-AUC
    pr_auc = average_precision_score(y_true, y_probs)
    print(f"PR-AUC: {pr_auc:.4f}")

    # 2. Performance at Matched FPR
    fpr_curve, tpr_curve, thresholds_roc = roc_curve(y_true, y_probs)
    idx = np.argmin(np.abs(fpr_curve - TARGET_FPR_MATCH))
    matched_th = thresholds_roc[idx]
    actual_fpr = fpr_curve[idx]

    matched_preds = (y_probs > matched_th).astype(int)
    cm = confusion_matrix(y_true, matched_preds)
    
    # Generate the full classification report
    report = classification_report(y_true, matched_preds, digits=4, target_names=["Benign", "Attack"], zero_division=0)
    print(f"\n--- Performance at Matched FPR ---")
    print(f"Target FPR: {TARGET_FPR_MATCH*100:.2f}% | Actual FPR Achieved: {actual_fpr*100:.2f}%")
    print(f"Matched Threshold: {matched_th:.4f}")
    # Print the full table instead of just the F1-score
    print("\nClassification Report:")
    print(report)
    print(f"Confusion Matrix:\n[TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}]\n[FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}]")

    # 3. Estimated Alarms per Hour
    fp = cm[0][1]
    alarms_per_hour = fp / TEST_SET_HOURS
    print(f"\n--- Operational Impact ---")
    print(f"Estimated Alarms/Hour (Based on {TEST_SET_HOURS} hours): {alarms_per_hour:.2f}")

    # 4. Save PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name.replace("_", " ")} (PR-AUC = {pr_auc:.4f})', color='darkred', lw=2)
    plt.xlabel('Recall (True Positive Rate)', fontsize=12)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
    plt.title(f'Precision-Recall Curve: {model_name.replace("_", " ")}', fontsize=14)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_filename = f"/home/spritz/storage/disk0/Master_Thesis/ReviewerImprovements/PR_Curve_{model_name}.png"
    plt.savefig(plot_filename, format='png', dpi=300)
    print(f"\n[+] Precision-Recall curve successfully saved as '{plot_filename}'.")

def evaluate_metric(df_val, df_test, metric_col, metric_name):
    """Calcola le soglie Unsupervised per una metrica specifica e testa i risultati."""
    print(f"\nRicerca della migliore soglia Unsupervised per: {metric_name.upper()}...")
    
    # 1. Pulizia dati per la colonna specifica
    val_scores = df_val[metric_col].fillna(df_val[metric_col].mean()).values
    test_scores = df_test[metric_col].fillna(df_test[metric_col].mean()).values
    y_test = df_test['True_Label'].values

    # 2. Garanzia Unsupervised: prendiamo solo i pacchetti sani dal Validation
    if 'True_Label' in df_val.columns and df_val['True_Label'].sum() > 0:
        val_sani = df_val[df_val['True_Label'] == 0][metric_col].values
    else:
        val_sani = val_scores

    best_f1 = -1
    best_fpr = 0
    best_th = 0
    best_preds = None

    for fpr in TARGET_FPRS:
        threshold = np.percentile(val_sani, fpr)
        preds = (test_scores < threshold).astype(int)
        
        current_f1 = f1_score(y_test, preds, zero_division=0)
        print(f" -> Tolleranza Val FPR: {fpr:>4.1f}% | Soglia: {threshold:>8.4f} | F1 Test: {current_f1:.4f}")
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_fpr = fpr
            best_th = threshold
            best_preds = preds

    # Invertiamo il segno degli score per calcolare l'AUC
    inverted_scores_test = -test_scores

    return {
        'name': metric_name,
        'best_f1': best_f1,
        'best_fpr': best_fpr,
        'best_th': best_th,
        'best_preds': best_preds,
        'inverted_scores': inverted_scores_test
    }

def main():
    print("Caricamento dataset Single Packet (Unsupervised)...")
    try:
        # Carichiamo le colonne essenziali
        df_val = pd.read_csv(VAL_FILE, usecols=['True_Label', 'Single_Score', 'Min_Single_Score'])
        df_test = pd.read_csv(TEST_FILE, usecols=['True_Label', 'Single_Score', 'Min_Single_Score'])
    except FileNotFoundError as e:
        print(f"Errore caricamento file: {e}")
        return
    except ValueError as e:
        print(f"Errore colonne: {e}\nAssicurati che il CSV contenga la colonna 'Min_Single_Score'.")
        return

    # --- ESECUZIONE DELLE DUE STRATEGIE ---
    res_mean = evaluate_metric(df_val, df_test, 'Single_Score', 'Media (Single_Score)')
    res_min = evaluate_metric(df_val, df_test, 'Min_Single_Score', 'Minimo (Min_Single_Score)')

    # --- STAMPA DEI DUE REPORT MIGLIORI ---
    print_report(
        df_test['True_Label'].values, 
        res_mean['best_preds'], 
        res_mean['inverted_scores'], 
        f"MIGLIOR RISULTATO: {res_mean['name'].upper()} (Ottimizzato per FPR: {res_mean['best_fpr']}%)"
    )

    print_report(
        df_test['True_Label'].values, 
        res_min['best_preds'], 
        res_min['inverted_scores'], 
        f"MIGLIOR RISULTATO: {res_min['name'].upper()} (Ottimizzato per FPR: {res_min['best_fpr']}%)"
    )

    # --- CLASSIFICA FINALE ---
    print("\n" + "="*75)
    print(" VINCITORE ASSOLUTO TRA LE DUE METRICHE")
    print("="*75)
    if res_min['best_f1'] > res_mean['best_f1']:
        print(f"🏆 Il MINIMO ha performato meglio (F1-Score: {res_min['best_f1']:.4f} vs {res_mean['best_f1']:.4f})")
    elif res_mean['best_f1'] > res_min['best_f1']:
        print(f"🏆 La MEDIA ha performato meglio (F1-Score: {res_mean['best_f1']:.4f} vs {res_min['best_f1']:.4f})")
    else:
        print(f"🤝 Pareggio tra MINIMO e MEDIA (F1-Score: {res_min['best_f1']:.4f})")

    # --- METRICHE AVANZATE E GRAFICI (NUOVA SEZIONE) ---
    print("\n" + "#"*75)
    print(" GENERAZIONE METRICHE AVANZATE E GRAFICI (CONCERNS 1 & 3)")
    print("#"*75)
    
    advanced_evaluation_for_reviewers(
        df_test['True_Label'].values, 
        res_mean['inverted_scores'], 
        model_name="Single_Packet_Mean"
    )
    
    advanced_evaluation_for_reviewers(
        df_test['True_Label'].values, 
        res_min['inverted_scores'], 
        model_name="Single_Packet_Min"
    )

if __name__ == "__main__":
    main()