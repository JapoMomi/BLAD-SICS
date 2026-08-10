import csv
import random

# --- CONFIGURATION ---
# Input files
NORMAL_FILE = "/home/spritz/storage/disk0/Master_Thesis/Dataset/normal_traffic.txt"
ATTACK_FILE = "/home/spritz/storage/disk0/Master_Thesis/Dataset/attack_traffic.txt"

# Output files
TRAIN_FILE = "train.txt"
VAL_FILE = "validation.txt"
TEST_FILE = "test.txt"

COL_IDX_TIME = 5 # L'indice del timestamp (in base alla tua struttura dati)

# -----------------------
# Step 1: Load normal and attack CSV rows
# -----------------------
def load_csv_rows(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 1:
                continue
            rows.append(row)
    return rows

print("Loading files...")
normal_rows = load_csv_rows(NORMAL_FILE)
attack_rows = load_csv_rows(ATTACK_FILE)

# -----------------------
# Step 2: Split normal rows into 80/10/10 (CORRETTO)
# -----------------------
# --- SHUFFLE BEFORE SPLITTING ---
print("Shuffling normal traffic to prevent Data Shift...")
random.seed(42)
random.shuffle(normal_rows) 
# ---------------------------------------------------------

n_total = len(normal_rows)
n_train = int(n_total * 0.8)
n_val = int(n_total * 0.1)
n_test_normal = n_total - n_train - n_val

train_rows = normal_rows[:n_train]
val_rows = normal_rows[n_train:n_train+n_val]
test_normal_rows = normal_rows[n_train+n_val:]

# -----------------------
# Step 3: Balance attack rows for test set
# -----------------------
max_attack_allowed = int(n_test_normal * 0.1) # Max 10% attacchi nel test set
n_test_attack = min(len(attack_rows), max_attack_allowed)
test_attack_rows = random.sample(attack_rows, n_test_attack)

# Combine normal + attack for test and shuffle AGAIN
test_rows = test_normal_rows + test_attack_rows
random.shuffle(test_rows)

# -----------------------
# Step 4: Write output CSV files
# -----------------------
def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

print("Writing splits...")
write_csv(f"/home/spritz/storage/disk0/Master_Thesis/Dataset/singlePacketSplits/{TRAIN_FILE}", train_rows)
write_csv(f"/home/spritz/storage/disk0/Master_Thesis/Dataset/singlePacketSplits/{VAL_FILE}", val_rows)
write_csv(f"/home/spritz/storage/disk0/Master_Thesis/Dataset/singlePacketSplits/{TEST_FILE}", test_rows)


# -----------------------
# Step 5: Calculate Test Set Duration
# -----------------------
try:
    # Estraiamo tutti i timestamp dal test set e li convertiamo in float
    timestamps = [float(row[COL_IDX_TIME]) for row in test_rows if len(row) > COL_IDX_TIME]
    
    if timestamps:
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        test_duration_seconds = max_ts - min_ts
        test_duration_hours = test_duration_seconds / 3600.0
    else:
        test_duration_seconds = 0.0
        test_duration_hours = 0.0
except Exception as e:
    print(f"Warning: Impossibile calcolare il tempo. Errore: {e}")
    test_duration_hours = 0.0

# --- STAMPA FINALE ---
print(f"\nFiles created successfully:")
print(f"  {TRAIN_FILE}: {len(train_rows)} rows")
print(f"  {VAL_FILE}: {len(val_rows)} rows")
print(f"  {TEST_FILE}: {len(test_rows)} rows ({len(test_normal_rows)} Benign + {len(test_attack_rows)} Attack)")
print(f"\n=============================================")
print(f"Durata Fisica del Test Set: {test_duration_hours:.2f} Ore")
print(f"=============================================")