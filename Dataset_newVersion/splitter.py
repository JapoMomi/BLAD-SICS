import pandas as pd
import numpy as np
import os

# --- USER CONFIGURATION ---
FULL_DATASET_FILE = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/IanRawDataset.txt" 
OUTPUT_DIR = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits"

SEQUENCE_LENGTH = 5
SEPARATOR = ' '
ATTACK_PERCENTAGE = 0.10  # Use 10% of available attacks in the Test Set

# CSV Column Indices
COL_IDX_PAYLOAD = 0
COL_IDX_LABEL1 = 1
COL_IDX_LABEL2 = 2
COL_IDX_TIME = 5

def load_and_sort(filepath):
    """Loads CSV and sorts it by Timestamp column."""
    print(f"Reading {filepath}...")
    df = pd.read_csv(filepath, header=None, dtype=str)
    df['ts_float'] = df[COL_IDX_TIME].astype(float)
    df = df.sort_values('ts_float')
    return df

def create_sliding_sequences(df, seq_len):
    """Creates sequences using a Sliding Window approach with Stride = 1."""
    sequences = []
    labels = []
    
    payloads = df[COL_IDX_PAYLOAD].tolist()
    # Determine label: 1 if attack, 0 if benign
    is_attack = ((df[COL_IDX_LABEL1] != '0') | (df[COL_IDX_LABEL2] != '0')).astype(int).tolist()
    
    # Sliding Window Loop
    for i in range(len(df) - seq_len + 1):
        seq_payloads = payloads[i : i+seq_len]
        seq_is_attack = is_attack[i : i+seq_len]
        
        full_seq = SEPARATOR.join(seq_payloads)
        label = 1 if max(seq_is_attack) > 0 else 0
        
        sequences.append(full_seq)
        labels.append(label)
        
    return sequences, labels

def save_to_csv(path, seqs, lbls):
    """Saves sequences to CSV with only 2 columns: payload, label."""
    df_out = pd.DataFrame({'payload': seqs, 'label': lbls})
    df_out.to_csv(path, header=False, index=False)
    print(f"Saved to {path}")

def print_stats(name, labels):
    """Prints count of Benign vs Attack sequences."""
    n_total = len(labels)
    n_attack = sum(labels)
    n_benign = n_total - n_attack
    print(f"--- {name} Stats ---")
    print(f"  Total Sequences: {n_total}")
    print(f"  Benign: {n_benign}")
    print(f"  Attack: {n_attack}")
    print("-----------------------")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load and Sort
    df_all = load_and_sort(FULL_DATASET_FILE)

    # 2. Filter Benign vs Attack
    mask_attack = (df_all[COL_IDX_LABEL1] != '0') | (df_all[COL_IDX_LABEL2] != '0')
    df_normal = df_all[~mask_attack].copy()
    df_attack = df_all[mask_attack].copy()

    # 3. Temporal Split (70/15/15)
    n = len(df_normal)
    idx_train_end = int(n * 0.70)
    idx_val_end = int(n * 0.85)

    df_train_raw = df_normal.iloc[:idx_train_end]
    df_val_raw = df_normal.iloc[idx_train_end:idx_val_end]
    df_test_base = df_normal.iloc[idx_val_end:]

    # 4. Prepare Test Set (Inject Attacks)
    n_attacks_to_use = int(len(df_attack) * ATTACK_PERCENTAGE)
    if n_attacks_to_use > 0:
        df_attack_subset = df_attack.sample(n=n_attacks_to_use, random_state=42)
        print(f"Injecting {len(df_attack_subset)} attacks into Test Set.")
        df_test_raw = pd.concat([df_test_base, df_attack_subset])
        df_test_raw = df_test_raw.sort_values('ts_float')
    else:
        df_test_raw = df_test_base

    # 5. Generate Sequences
    print("\nGenerating sequences...")
    train_seqs, train_lbls = create_sliding_sequences(df_train_raw, SEQUENCE_LENGTH)
    val_seqs, val_lbls = create_sliding_sequences(df_val_raw, SEQUENCE_LENGTH)
    test_seqs, test_lbls = create_sliding_sequences(df_test_raw, SEQUENCE_LENGTH)

    # 6. Print Stats (Requested Feature)
    print_stats("TRAIN SET", train_lbls)
    print_stats("VALIDATION SET", val_lbls)
    print_stats("TEST SET", test_lbls)

    # 7. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_to_csv(os.path.join(OUTPUT_DIR, "train.txt"), train_seqs, train_lbls)
    save_to_csv(os.path.join(OUTPUT_DIR, "validation.txt"), val_seqs, val_lbls)
    save_to_csv(os.path.join(OUTPUT_DIR, "test.txt"), test_seqs, test_lbls)