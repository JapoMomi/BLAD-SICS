import pandas as pd
import numpy as np

# --- CONFIGURAZIONE PERCORSI ---
# Assicurati che il nome del file coincida con quello generato dal tuo script di detection
INPUT_CSV = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/detection_detailed_results.csv"
OUTPUT_CSV = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/analysis_false_negatives.csv"

def analyze_false_negatives():
    print(f"--- Caricamento risultati da: {INPUT_CSV} ---")
    
    try:
        # Carica il dataset
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"❌ Errore: Il file {INPUT_CSV} non è stato trovato.")
        return

    # --- FILTRO ---
    # Label == 1 (È un Attacco)
    # Pred == 0  (Il modello ha detto Benigno)
    fn_df = df[(df['Label'] == 1) & (df['Pred'] == 0)].copy()
    
    count_total = len(df)
    count_fn = len(fn_df)
    
    print(f"Totale righe nel report: {count_total}")
    print(f"Falsi Negativi trovati (Label=1, Pred=0): {count_fn}")
    
    if count_fn == 0:
        print("✅ Ottimo! Nessun Falso Negativo trovato.")
        return

    # --- ANALISI RAPIDA ---
    print("\n--- Statistiche degli Score dei Falsi Negativi ---")
    # Vediamo quanto erano alti i punteggi (ricorda: punteggi alti = sembrano benigni)
    stats = fn_df['Avg_Score'].describe()
    print(stats)
    
    # Salvataggio su file
    print(f"\nSalvataggio delle righe problematiche in: {OUTPUT_CSV}")
    fn_df.to_csv(OUTPUT_CSV, index=False)
    
    # --- ANTEPRIMA ---
    print("\nEsempio delle prime 5 righe problematiche:")
    # Mostriamo solo le colonne più utili
    cols_to_show = ['Label', 'Pred', 'Avg_Score', 'Score_P1', 'Score_P2', 'Score_P3', 'Score_P4', 'Score_P5']
    # Se esistono nel csv, altrimenti mostra tutto
    available_cols = [c for c in cols_to_show if c in fn_df.columns]
    print(fn_df[available_cols].head(5))

    print("\n--- SUGGERIMENTO PER L'ANALISI ---")
    print("Controlla la colonna 'Avg_Score' nel file salvato.")
    print("1. Se i valori sono VICINI alla soglia (es. -0.70 se la soglia è -0.75), allora bastava poco per prenderli.")
    print("2. Se i valori sono MOLTO ALTI (es. -0.001), il modello è completamente convinto che siano benigni.")

if __name__ == "__main__":
    analyze_false_negatives()