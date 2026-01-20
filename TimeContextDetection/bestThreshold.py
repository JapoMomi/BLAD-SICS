import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# --- CONFIGURAZIONE ---
INPUT_FILE = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/detection_detailed_results.csv"
OUTPUT_THRESHOLDS_FILE = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/best_thresholds_found.txt"

def load_data():
    print(f"--- Caricamento risultati da {INPUT_FILE} ---")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("❌ Errore: File non trovato.")
        return None

    if 'Min_Score' not in df.columns:
        score_cols = [c for c in df.columns if c.startswith('Score_P')]
        if score_cols:
            print("⚠️ Colonna 'Min_Score' mancante. Calcolo al volo...")
            df['Min_Score'] = df[score_cols].min(axis=1)
        else:
            print("❌ Errore: Impossibile calcolare Min_Score.")
            return None
    return df

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "f1": f1, "recall": recall, "precision": precision, 
        "fpr": fpr, "specificity": specificity
    }

def print_strategy_header(name, thresholds=None):
    print(f"\n{'='*80}")
    print(f"STRATEGIA: {name.upper()}")
    if thresholds:
        print(f"Soglie Trovate -> Avg: {thresholds[0]:.4f} | Min: {thresholds[1]:.4f}")
    print(f"{'-'*80}")

def run_optimization():
    df = load_data()
    if df is None: return

    y_true = df['Label'].values
    avg_scores = df['Avg_Score'].values
    min_scores = df['Min_Score'].values
    
    # --- 1. SETUP GRID SEARCH ---
    avg_candidates = np.unique(np.percentile(avg_scores, np.linspace(0.1, 40, 40)))
    min_candidates = np.unique(np.percentile(min_scores, np.linspace(0.1, 30, 40)))
    
    total_combs = len(avg_candidates) * len(min_candidates)
    print(f"🔍 Testando {total_combs} combinazioni per 4 strategie diverse...")

    best_results = {
        "f1_max": {"score": -1, "thresh": (0,0), "preds": None},
        "roc_best": {"score": -1, "thresh": (0,0), "preds": None}, 
        "min_fp": {"score": -1, "thresh": (0,0), "preds": None},   
        "min_fn": {"score": -1, "thresh": (0,0), "preds": None}    
    }

    # --- 2. GRID SEARCH LOOP ---
    for t_avg in avg_candidates:
        for t_min in min_candidates:
            y_pred = ((avg_scores < t_avg) | (min_scores < t_min)).astype(int)
            
            # Calcolo metriche vettoriali
            tp = np.sum((y_pred == 1) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            specificity = 1 - fpr
            
            # Strategia 1: Max F1
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_results["f1_max"]["score"]:
                best_results["f1_max"]["score"] = f1
                best_results["f1_max"]["thresh"] = (t_avg, t_min)
                best_results["f1_max"]["preds"] = y_pred

            # Strategia 2: Max ROC (Youden)
            youden = recall + specificity - 1
            if youden > best_results["roc_best"]["score"]:
                best_results["roc_best"]["score"] = youden
                best_results["roc_best"]["thresh"] = (t_avg, t_min)
                best_results["roc_best"]["preds"] = y_pred

            # Strategia 3: Minimize FP (Conservative)
            # FPR < 0.5%
            if fpr <= 0.005: 
                score_conservative = recall
            else:
                score_conservative = -1 
            
            if score_conservative > best_results["min_fp"]["score"]:
                best_results["min_fp"]["score"] = score_conservative
                best_results["min_fp"]["thresh"] = (t_avg, t_min)
                best_results["min_fp"]["preds"] = y_pred

            # Strategia 4: Minimize FN (Paranoid)
            score_paranoid = (recall * 10) + precision
            if score_paranoid > best_results["min_fn"]["score"]:
                best_results["min_fn"]["score"] = score_paranoid
                best_results["min_fn"]["thresh"] = (t_avg, t_min)
                best_results["min_fn"]["preds"] = y_pred

    # --- 3. STAMPA REPORT DETTAGLIATI ---
    strategies_metrics = {}
    
    # Report per la Originale (CSV attuale)
    if 'Pred' in df.columns:
        print_strategy_header("Original (Current CSV)", None)
        # Stampa il Classification Report
        print(classification_report(y_true, df['Pred'].values, target_names=["Benign", "Attack"]))
        # Calcola metriche per riassunto finale
        metrics = calculate_metrics(y_true, df['Pred'].values)
        strategies_metrics["Original"] = metrics
        print(f"Confusion Matrix: [TP: {metrics['tp']} | FN: {metrics['fn']}]")
        print(f"                  [FP: {metrics['fp']} | TN: {metrics['tn']}]")


    # Report per le 4 Strategie Ottimizzate
    for name, res in best_results.items():
        if res["preds"] is not None:
            metrics = calculate_metrics(y_true, res["preds"])
            strategies_metrics[name] = metrics
            
            print_strategy_header(name, res["thresh"])
            
            # --- QUI STAMPIAMO IL GRAFICO (REPORT) CHE CHIEDEVI ---
            print(classification_report(y_true, res["preds"], target_names=["Benign", "Attack"]))
            
            print(f"Confusion Matrix: [TP: {metrics['tp']} | FN: {metrics['fn']}]")
            print(f"                  [FP: {metrics['fp']} | TN: {metrics['tn']}]")
            print(f"Metrics Extra:    FPR: {metrics['fpr']:.2%}")

    # --- 4. CONFRONTO FINALE ---
    print("\n\n🏆 CONFRONTO FINALE STRATEGIE 🏆")
    print(f"{'Strategia':<20} | {'F1-Score':<8} | {'Recall':<8} | {'FPR':<8} | {'FP Count':<8} | {'FN Count':<8}")
    print("-" * 80)
    
    best_overall_name = "f1_max"
    
    for name, m in strategies_metrics.items():
        print(f"{name:<20} | {m['f1']:.4f}   | {m['recall']:.4f}   | {m['fpr']:.2%}   | {m['fp']:<8} | {m['fn']:<8}")

    print("-" * 80)
    
    rec = strategies_metrics.get("min_fp")
    f1_rec = strategies_metrics.get("f1_max")
    
    print("\n💡 RACCOMANDAZIONE:")
    if rec and rec['recall'] > 0.85:
        print("✅ VINCITORE: 'MIN_FP' (Conservative).")
        print("   Motivo: Offre una Recall eccellente (>85%) con Falsi Allarmi quasi nulli.")
        best_overall_name = "min_fp"
    elif f1_rec:
        print("✅ VINCITORE: 'F1_MAX' (Balanced).")
        print("   Motivo: Miglior compromesso matematico.")
        best_overall_name = "f1_max"
        
    winner = best_results.get(best_overall_name)
    if winner:
        t_avg, t_min = winner["thresh"]
        with open(OUTPUT_THRESHOLDS_FILE, "w") as f:
            f.write(f"AVG_THRESH={t_avg}\n")
            f.write(f"MIN_THRESH={t_min}\n")
            f.write(f"STRATEGY={best_overall_name}\n")
        print(f"\n💾 Soglie della strategia vincente ({best_overall_name}) salvate in: {OUTPUT_THRESHOLDS_FILE}")

if __name__ == "__main__":
    run_optimization()