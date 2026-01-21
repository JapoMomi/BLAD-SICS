import pandas as pd
import numpy as np
import os

# --- USER CONFIGURATION ---
# Assicurati che questi percorsi siano corretti per il tuo sistema
FULL_DATASET_FILE = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/IanRawDataset.txt" 
OUTPUT_DIR = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits"

SEQUENCE_LENGTH = 5
SEPARATOR = ' '
ATTACK_PERCENTAGE = 0.20  # Use 10% of available attacks in the Test Set

# CSV Column Indices
COL_IDX_PAYLOAD = 0
COL_IDX_LABEL1 = 1
COL_IDX_LABEL2 = 2
COL_IDX_TIME = 5

def load_and_sort(filepath):
    """Loads CSV and sorts it by Timestamp column."""
    print(f"Reading {filepath}...")
    df = pd.read_csv(filepath, header=None, dtype=str)
    # Convert timestamp to float for sorting
    df['ts_float'] = df[COL_IDX_TIME].astype(float)
    df = df.sort_values('ts_float')
    return df

def create_sliding_sequences_detailed(df, seq_len):
    """
    Creates sequences using a Sliding Window approach (Stride = 1).
    
    FORMAT:
    Payloads (space separated), Label_P1, Label_P2, ..., Label_P5, Sequence_Label
    
    Example:
    "04ab... 05cd...", 0, 0, 1, 0, 0, 1
    """
    sequences = []
    labels_string_list = []
    
    # 1. Extract Payloads and Individual Packet Labels
    payloads = df[COL_IDX_PAYLOAD].tolist()
    
    # Pre-calculate boolean mask: 1 if row is attack, 0 if benign
    # Checks if either label column is NOT '0'
    row_is_attack = ((df[COL_IDX_LABEL1] != '0') | (df[COL_IDX_LABEL2] != '0')).astype(int).tolist()
    
    num_sequences = len(payloads) - seq_len + 1
    
    print(f"  Processing {num_sequences} windows...")

    for i in range(num_sequences):
        # A. Create Sequence String
        window_payloads = payloads[i : i + seq_len]
        seq_str = SEPARATOR.join(window_payloads)
        
        # B. Get Individual Labels for this window
        window_packet_labels = row_is_attack[i : i + seq_len] # List of [0, 1, 0...]
        
        # C. Determine Sequence Label (1 if ANY packet is attack)
        seq_label = 1 if sum(window_packet_labels) > 0 else 0
        
        # D. Create the Label String Part
        # Format: L1,L2,L3,L4,L5,SeqL
        # Join individual labels with comma
        packet_labels_str = ",".join(map(str, window_packet_labels))
        
        # Combine everything
        full_label_str = f"{packet_labels_str},{seq_label}"
        
        sequences.append(seq_str)
        labels_string_list.append(full_label_str)
        
    return sequences, labels_string_list

def save_to_txt(sequences, labels, output_path):
    print(f"Saving {len(sequences)} sequences to {output_path}...")
    with open(output_path, "w") as f:
        for seq, lbl in zip(sequences, labels):
            # Format: <PAYLOADS>,<L1>,<L2>,<L3>,<L4>,<L5>,<SEQ_L>
            f.write(f"{seq},{lbl}\n")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Data
    df = load_and_sort(FULL_DATASET_FILE)
    
    # 2. Separate Benign and Attack
    # Check if row is benign (both labels are '0')
    is_benign = (df[COL_IDX_LABEL1] == '0') & (df[COL_IDX_LABEL2] == '0')
    
    df_normal = df[is_benign]
    df_attack = df[~is_benign]

    print(f"Total Normal: {len(df_normal)}")
    print(f"Total Attack: {len(df_attack)}")

    # 3. Time-Split Normal Data (Train / Validation / Test_Base)
    # Using 70% Train, 15% Val, 15% Test
    n = len(df_normal)
    idx_train_end = int(n * 0.70)
    idx_val_end = int(n * 0.85)

    df_train_raw = df_normal.iloc[:idx_train_end]
    df_val_raw = df_normal.iloc[idx_train_end:idx_val_end]
    df_test_base = df_normal.iloc[idx_val_end:]

    # 4. Prepare Test Set (Inject Attacks)
    n_attacks_to_use = int(len(df_attack) * ATTACK_PERCENTAGE)
    
    if n_attacks_to_use > 0:
        # Sample attacks randomly to mix them in
        df_attack_subset = df_attack.sample(n=n_attacks_to_use, random_state=42)
        print(f"Injecting {len(df_attack_subset)} attacks into Test Set.")
        
        # Combine Normal Test + Attacks and Re-Sort by Time
        # This simulates a real timeline where attacks happen amidst normal traffic
        df_test_raw = pd.concat([df_test_base, df_attack_subset])
        df_test_raw = df_test_raw.sort_values('ts_float')
    else:
        df_test_raw = df_test_base

    # 5. Generate Sequences with DETAILED LABELS
    print("\nGenerating sequences...")
    
    # Train
    train_seqs, train_lbls = create_sliding_sequences_detailed(df_train_raw, SEQUENCE_LENGTH)
    save_to_txt(train_seqs, train_lbls, os.path.join(OUTPUT_DIR, "train.txt"))

    # Validation
    val_seqs, val_lbls = create_sliding_sequences_detailed(df_val_raw, SEQUENCE_LENGTH)
    save_to_txt(val_seqs, val_lbls, os.path.join(OUTPUT_DIR, "validation.txt"))

    # Test
    test_seqs, test_lbls = create_sliding_sequences_detailed(df_test_raw, SEQUENCE_LENGTH)
    save_to_txt(test_seqs, test_lbls, os.path.join(OUTPUT_DIR, "test.txt"))

    print("\nSplitting and Generation Complete.")
    print(f"Output files located in: {OUTPUT_DIR}")