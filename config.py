# --------------------
# Dataset Configuration
# --------------------
DATA_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset/splits"

N_PACKETS = 3           # Number of packets per sequence
MAX_TIME_GAP = 2        # Seconds between packets
MAX_LENGTH = 768        # Tokenizer sequence length limit

# --------------------
# Model Configuration
# --------------------
MODEL_NAME = "google/byt5-small"  # or "google/byt5-small"

# --------------------
# Training Configuration
# --------------------
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_EVAL = 8
WEIGHT_DECAY = 0.08
GRADIENT_ACCUMULATION = 8
FP16 = False
WARMUP_RATIO = 0.08
LR_SCHEDULER = "cosine_with_restarts"

# --------------------
# Output & Logging
# --------------------
OUTPUT_DIR = "/home/spritz/storage/disk0/Master_Thesis/ByT5/ByT5-project-sequences"
LOGS_DIR = f"{OUTPUT_DIR}/logs"
WANDB_DISABLED = True  # Disable Weights & Biases

# --------------------
# Detection Configuration
# --------------------
BATCH_SIZE = 32
LAST_N_LAYERS = 3
PCA_DIM = 512          # compressed embedding size