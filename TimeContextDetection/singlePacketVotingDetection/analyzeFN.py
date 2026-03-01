import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
INPUT_CSV = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/detection_final_with_preds.csv"
OUTPUT_FN_CSV = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/false_negatives_analysis.csv"

# INSERISCI QUI LA SOGLIA CHE HA VINTO NELLO SCRIPT PRECEDENTE!
# Esempio: se il report diceva "Soglia: -2.4668", metti -2.4668
THRESHOLD_USED = -2.1666 

def main():
    print(f"Caricamento dati da {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print("❌ File non trovato.")
        return

    # 1. Identificazione dei Veri Positivi e Falsi Negativi
    attacchi_totali = df[df['True_Label'] == 1]
    tp_df = df[(df['True_Label'] == 1) & (df['Voting_Pred'] == 1)]
    fn_df = df[(df['True_Label'] == 1) & (df['Voting_Pred'] == 0)].copy()

    num_attacchi = len(attacchi_totali)
    num_tp = len(tp_df)
    num_fn = len(fn_df)

    print(f"\n{'='*60}")
    print(f" RESOCONTO ATTACCHI")
    print(f"{'='*60}")
    print(f"Attacchi totali nel dataset: {num_attacchi}")
    print(f"Rilevati (True Positives):   {num_tp} ({num_tp/num_attacchi:.1%})")
    print(f"Mancati  (False Negatives):  {num_fn} ({num_fn/num_attacchi:.1%})")

    if num_fn == 0:
        print("🎉 Nessun Falso Negativo! Il modello è perfetto.")
        return

    # 2. Analisi approfondita dei Falsi Negativi
    score_cols = [f'LogProb_Pos{i}' for i in range(5)]
    
    # Calcoliamo la media delle log-probabilità per ogni FN
    fn_df['Avg_LogProb'] = fn_df[score_cols].mean(axis=1)
    
    # Contiamo QUANTE VOLTE il pacchetto è sceso sotto la soglia (cioè quanti voti di anomalia ha preso)
    # is_anomalous_mask sarà True dove LogProb < THRESHOLD_USED
    is_anomalous_mask = fn_df[score_cols] < THRESHOLD_USED
    fn_df['Voti_Anomalia'] = is_anomalous_mask.sum(axis=1)

    print(f"\n{'='*60}")
    print(f" ANALISI DEI FALSI NEGATIVI (FN)")
    print(f"{'='*60}")

    # Statistiche sui "Quasi-Rilevati"
    voti_counts = fn_df['Voti_Anomalia'].value_counts().sort_index()
    print("Distribuzione dei voti di anomalia tra i FN:")
    for voti, count in voti_counts.items():
        print(f" - FN con {voti} voti di anomalia: {count} pacchetti ({count/num_fn:.1%})")

    # Confronto delle medie
    avg_tp_score = tp_df[score_cols].mean(axis=1).mean()
    avg_fn_score = fn_df['Avg_LogProb'].mean()
    print(f"\nLog-Probabilità Media dei Veri Positivi: {avg_tp_score:.4f} (Ben sotto la soglia)")
    print(f"Log-Probabilità Media dei Falsi Negativi: {avg_fn_score:.4f} (Troppo vicina a 0)")

    # 3. Ordinamento e Salvataggio
    # Ordiniamo i FN per 'Avg_LogProb' decrescente (dal più vicino a 0 al più negativo)
    # Quelli in cima alla lista sono quelli che il modello ha scambiato per "Normalissimi"
    fn_df = fn_df.sort_values(by='Avg_LogProb', ascending=False)
    
    # Selezioniamo solo le colonne utili per il CSV di output
    cols_to_save = ['Packet_ID', 'Voti_Anomalia', 'Avg_LogProb'] + score_cols
    fn_df[cols_to_save].to_csv(OUTPUT_FN_CSV, index=False)

    print(f"\n✅ Dettaglio completo dei Falsi Negativi salvato in: {OUTPUT_FN_CSV}")

if __name__ == "__main__":
    main()