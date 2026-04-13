import pandas as pd
import numpy as np
import os

# --- USER CONFIGURATION ---
FULL_DATASET_FILE = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/IanRawDataset.txt" 
OUTPUT_DIR = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits"

SEQUENCE_LENGTH = 5
SEPARATOR = ' '

# CSV Column Indices
COL_IDX_PAYLOAD = 0
COL_IDX_LABEL1 = 1
COL_IDX_LABEL2 = 2
COL_IDX_TIME = 5

def load_and_sort(filepath):
    print(f"Reading {filepath}...")
    df = pd.read_csv(filepath, header=None, dtype=str)
    df['ts_float'] = df[COL_IDX_TIME].astype(float)
    df = df.sort_values('ts_float')
    return df

def create_sequences(df, seq_len):
    """
    Crea le sequenze dal DataFrame fornito.
    Se il DF è stato pre-filtrato (solo benigni), le sequenze salteranno gli attacchi
    unendo i pacchetti benigni adiacenti.
    """
    sequences = []
    labels_string_list = []
    
    payloads = df[COL_IDX_PAYLOAD].tolist()
    
    # Maschera: 1 se attacco, 0 se benigno
    row_is_attack = ((df[COL_IDX_LABEL1] != '0') | (df[COL_IDX_LABEL2] != '0')).astype(int).tolist()
    
    num_sequences = len(payloads) - seq_len + 1

    for i in range(num_sequences):
        window_payloads = payloads[i : i + seq_len]
        seq_str = SEPARATOR.join(window_payloads)
        
        window_packet_labels = row_is_attack[i : i + seq_len]
        
        # Etichetta sequenza (1 se c'è almeno un attacco)
        seq_label = 1 if sum(window_packet_labels) > 0 else 0
        
        # Formato: L1,L2,L3,L4,L5,SeqL
        packet_labels_str = ",".join(map(str, window_packet_labels))
        full_label_str = f"{packet_labels_str},{seq_label}"
        
        sequences.append(seq_str)
        labels_string_list.append(full_label_str)
        
    return sequences, labels_string_list

def save_to_txt(sequences, labels, output_path):
    print(f"Saving {len(sequences)} sequences to {output_path}...")
    with open(output_path, "w") as f:
        for seq, lbl in zip(sequences, labels):
            f.write(f"{seq},{lbl}\n")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Carica e Ordina TUTTO il dataset cronologicamente
    df_full = load_and_sort(FULL_DATASET_FILE)
    
    # 2. Calcola i punti di taglio per lo split temporale (70% - 15% - 15%)
    total_rows = len(df_full)
    idx_train_end = int(total_rows * 0.70)
    idx_val_end = int(total_rows * 0.85)

    # 3. Dividi il DataFrame grezzo in base al tempo
    df_train_raw = df_full.iloc[:idx_train_end]
    df_val_raw = df_full.iloc[idx_train_end:idx_val_end]
    df_test = df_full.iloc[idx_val_end:] # Il Test rimane intatto

    # 4. Filtra GLI ATTACCHI da Train e Val (Tieni solo il traffico benigno)
    cond_benign_train = (df_train_raw[COL_IDX_LABEL1] == '0') & (df_train_raw[COL_IDX_LABEL2] == '0')
    cond_benign_val = (df_val_raw[COL_IDX_LABEL1] == '0') & (df_val_raw[COL_IDX_LABEL2] == '0')

    df_train_benign = df_train_raw[cond_benign_train]
    df_val_benign = df_val_raw[cond_benign_val]

    print(f"\n--- Statistiche pre-generazione (su righe/pacchetti) ---")
    print(f"Train Raw: {len(df_train_raw)} -> Train Benign (Filtrato): {len(df_train_benign)} pacchetti")
    print(f"Val Raw: {len(df_val_raw)} -> Val Benign (Filtrato): {len(df_val_benign)} pacchetti")
    print(f"Test Raw (Intatto): {len(df_test)} pacchetti (include attacchi)")

    # 5. Genera le sequenze
    print("\nGenerazione sequenze TRAIN (solo benigni)...")
    train_seqs, train_lbls = create_sequences(df_train_benign, SEQUENCE_LENGTH)

    print("Generazione sequenze VALIDATION (solo benigni)...")
    val_seqs, val_lbls = create_sequences(df_val_benign, SEQUENCE_LENGTH)

    print("Generazione sequenze TEST (reale, include attacchi)...")
    test_seqs, test_lbls = create_sequences(df_test, SEQUENCE_LENGTH)

    # --- 6. ANALISI E CONTEGGIO DEL TEST SET ---
    # Una stringa di label finisce con ',1' se la sequenza è anomala, e con ',0' se è normale.
    test_anomalous_count = sum(1 for lbl in test_lbls if lbl.endswith(',1'))
    test_normal_count = len(test_lbls) - test_anomalous_count
    total_test_seqs = len(test_lbls)

    print("\n=============================================")
    print("      ANALISI FINALE DEL TEST SET            ")
    print("=============================================")
    print(f"Sequenze Normali: {test_normal_count} ({ (test_normal_count/total_test_seqs)*100:.2f}% )")
    print(f"Sequenze Anomale: {test_anomalous_count} ({ (test_anomalous_count/total_test_seqs)*100:.2f}% )")
    print("=============================================\n")

    # 7. Salvataggio
    save_to_txt(train_seqs, train_lbls, os.path.join(OUTPUT_DIR, "train.txt"))
    save_to_txt(val_seqs, val_lbls, os.path.join(OUTPUT_DIR, "validation.txt"))
    save_to_txt(test_seqs, test_lbls, os.path.join(OUTPUT_DIR, "test.txt"))

    print("Splitting e Generazione Completati con successo.")