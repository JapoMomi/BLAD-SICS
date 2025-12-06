import pandas as pd
import random
import time

# --- Configuration ---
NUM_TRAIN = 2000
NUM_VAL = 750
NUM_TEST_NORMAL = 525
NUM_TEST_ATTACK = 75

# --- Vocabulary ---
# Normal: Standard letters
VOCAB_NORMAL = list("abcdefghilmnopqrstuvz")

# Attack: Special letters and symbols (The contamination)
VOCAB_ATTACK = list("wxyjk#@-!?")

def to_hex(text):
    """Encodes text to hex string."""
    return text.encode('latin-1').hex()

def generate_packet(is_attack=False):
    """
    Generates a packet.
    If Attack: Contains mostly normal chars mixed with 10-20% attack chars.
    """
    length = random.randint(20, 60)
    
    if is_attack:
        # Determine how many "bad" characters to inject (1% to 15%)
        ratio = random.uniform(0.05, 0.20)
        n_attack = int(length * ratio)
        
        # Ensure at least 1 anomaly exists if it's an attack
        if n_attack == 0:
            n_attack = 1
            
        n_normal = length - n_attack
        
        # Select characters
        chars_attack = random.choices(VOCAB_ATTACK, k=n_attack)
        chars_normal = random.choices(VOCAB_NORMAL, k=n_normal)
        
        # Mix them together and shuffle
        packet_chars = chars_attack + chars_normal
        random.shuffle(packet_chars)
        content = "".join(packet_chars)
        
        # Labels: 1 for Attack
        cat = 1
        type_ = 1
    else:
        # Normal: 100% normal characters
        content = "".join(random.choices(VOCAB_NORMAL, k=length))
        
        # Labels: 0 for Normal
        cat = 0
        type_ = 0
        
    # Convert payload to Hex
    payload_hex = to_hex(content)
    #payload_hex = content
    # Random Source and Destination (1-4)
    src = random.choice([1, 2, 3, 4])
    dst = random.choice([1, 2, 3, 4])
    while dst == src: # Basic rule: don't send to self
        dst = random.choice([1, 2, 3, 4])
        
    # Generate a Timestamp
    timestamp = time.time() + random.uniform(0, 10000)
    
    return [payload_hex, cat, type_, src, dst, timestamp]

# --- Main Execution ---
print("Generating 'Mixed' Dataset...")

# 1. Train Set (Normal Only)
train_data = [generate_packet(is_attack=False) for _ in range(NUM_TRAIN)]
df_train = pd.DataFrame(train_data)
df_train.to_csv('/home/spritz/storage/disk0/Master_Thesis/Dataset/simplified_dataset/simple_mixed_train.csv', header=False, index=False)
print(f"Created simple_mixed_train.csv ({len(df_train)} samples)")

# 2. Validation Set (Normal Only)
val_data = [generate_packet(is_attack=False) for _ in range(NUM_VAL)]
df_val = pd.DataFrame(val_data)
df_val.to_csv('/home/spritz/storage/disk0/Master_Thesis/Dataset/simplified_dataset/simple_mixed_val.csv', header=False, index=False)
print(f"Created simple_mixed_val.csv ({len(df_val)} samples)")

# 3. Test Set (Mixed Normal & Attack)
test_data = []
test_data.extend([generate_packet(is_attack=False) for _ in range(NUM_TEST_NORMAL)])
test_data.extend([generate_packet(is_attack=True) for _ in range(NUM_TEST_ATTACK)])
random.shuffle(test_data)

df_test = pd.DataFrame(test_data)
df_test.to_csv('/home/spritz/storage/disk0/Master_Thesis/Dataset/simplified_dataset/simple_mixed_test.csv', header=False, index=False)
print(f"Created simple_mixed_test.csv ({len(df_test)} mixed samples)")