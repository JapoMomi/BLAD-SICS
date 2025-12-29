import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import os

# --- 1. CONFIGURATION ---
SEQUENCE_LENGTH = 5        
MAX_LENGTH = 512         
PACKET_SEPARATOR = ' '      
# UPDATE THIS PATH to point to the model created by training.py
MODEL_PATH = f"/home/spritz/storage/disk0/Master_Thesis/Stuff/Byt5/BYTES_modbus-sequence_{SEQUENCE_LENGTH}-finetuned" 
VAL_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset/splits/validation.txt"
TEST_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset/splits/test.txt"

# --- 2. Data Loading ---
def load_labeled_context_data(filepath, N, separator):
    df = pd.read_csv(
        filepath, 
        header=None, 
        usecols=[0, 1, 2], 
        names=['payload', 'cat', 'type'],
        dtype={'payload': str}, 
        keep_default_na=False
    )
    
    # Label each individual packet (1 if Attack, 0 if Benign)
    is_attack_labels = ((df['cat'] != 0) | (df['type'] != 0)).astype(int).tolist()
    
    processed_sequences = []
    final_sequence_labels = []
    
    # Group into sequences of N
    for i in range(0, len(df), N):
        chunk = df.iloc[i:i + N]
        if len(chunk) < N: continue

        # Join Hex payloads (We convert to Bytes later in Dataset)
        sequence_payload = separator.join(chunk['payload'].tolist())
        
        # Sequence label: 1 if ANY packet in chunk is an attack
        chunk_labels = is_attack_labels[i:i+N]
        sequence_label = 1 if max(chunk_labels) > 0 else 0
        
        processed_sequences.append(sequence_payload)
        final_sequence_labels.append(sequence_label)

    return processed_sequences, final_sequence_labels

# --- 3. Dataset Class (NO MASKING HERE) ---
class ModbusDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        hex_text = self.texts[idx]
        
        # --- HEX TO BYTES CONVERSION (Matching Training) ---
        packet_list = hex_text.split(PACKET_SEPARATOR)
        raw_content_list = []
        for hex_segment in packet_list:
            try:
                # Decode hex to bytes -> Latin-1 string
                raw_content_list.append(bytes.fromhex(hex_segment).decode('latin-1'))
            except:
                raw_content_list.append(hex_segment)
                 
        raw_content = PACKET_SEPARATOR.join(raw_content_list)
        
        # Tokenize (Clean Input)
        encoding = self.tokenizer(
            raw_content, 
            max_length=self.max_length,
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        
        # Return clean input. Labels = Input for calculation
        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

# --- 4. Sliding Window Scoring Function ---
def get_anomaly_scores(model, dataset, device, tokenizer):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    
    # Lists to store the stats for each sequence
    seq_means = []
    seq_mins = []
    seq_medians = []

    # ByT5 Sentinel Token ID for <extra_id_0>
    mask_token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    
    print(f"Scoring {len(dataset)} sequences using Sliding Window Masking...")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 50 == 0: print(f"Processing sample {i}...")
            
            original_input_ids = batch['input_ids'].to(device) # [1, SeqLen]
            labels = batch['labels'].to(device)
            
            # Find actual length (ignore padding)
            non_pad = (original_input_ids != tokenizer.pad_token_id).sum().item()
            
            # Window Setup
            window_size = 5  # Size of mask
            stride = 2       # Step size
            
            chunk_scores = []
            
            # Slide over the valid content
            for j in range(0, non_pad, stride):
                masked_input = original_input_ids.clone()
                end = min(j + window_size, non_pad)
                
                # Apply Mask to window
                masked_input[0, j:end] = mask_token_id
                
                # Forward Pass
                outputs = model(input_ids=masked_input, labels=labels)
                log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
                
                # Get probability of TRUE token
                target_ids = labels.unsqueeze(-1)
                token_log_probs = log_probs.gather(-1, target_ids).squeeze(-1)
                
                # Score = Average LogProb of the MASKED window only
                window_log_probs = token_log_probs[0, j:end]
                score = window_log_probs.mean().item()
                chunk_scores.append(score)
            
            # --- Store Stats ---
            if chunk_scores:
                seq_means.append(np.mean(chunk_scores))
                seq_mins.append(np.min(chunk_scores))
                seq_medians.append(np.median(chunk_scores))
            else:
                # Fallback for empty/failed sequences
                seq_means.append(-100)
                seq_mins.append(-100)
                seq_medians.append(-100)

    # Return a dictionary of arrays
    return {
        "mean": np.array(seq_means),
        "min": np.array(seq_mins),
        "median": np.array(seq_medians)
    }

# --- 5. Main Execution ---
if __name__ == "__main__":
    print(f"Starting Sequence Detection (N={SEQUENCE_LENGTH})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading Model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True).to(device)

    # Load Data
    val_texts, val_labels = load_labeled_context_data(VAL_PATH, SEQUENCE_LENGTH, PACKET_SEPARATOR)
    test_texts, test_labels = load_labeled_context_data(TEST_PATH, SEQUENCE_LENGTH, PACKET_SEPARATOR)

    val_dataset = ModbusDataset(val_texts, tokenizer, max_length=MAX_LENGTH)
    test_dataset = ModbusDataset(test_texts, tokenizer, max_length=MAX_LENGTH)

    # 1. Validation (Benign) Scores
    print("\n--- Calculating Validation Threshold ---")
    val_results = get_anomaly_scores(model, val_dataset, device, tokenizer)
    
    # We use the MEAN score to determine the threshold (consistent with previous logic)
    val_scores_mean = val_results["median"]
    
    # Threshold = 2nd percentile (Adjusted for Real Data Noise)
    prob_threshold = np.percentile(val_scores_mean, 2) 

    print(f"Validation Mean Stats -> Mean: {val_scores_mean.mean():.4f}, Min: {val_scores_mean.min():.4f}")
    print(f"Selected Threshold (2th percentile of Mean scores): {prob_threshold:.4f}")

    # 2. Test Scores
    print("\n--- Evaluating Test Set ---")
    test_results = get_anomaly_scores(model, test_dataset, device, tokenizer)
    
    # Extract arrays
    test_means = test_results["mean"]
    test_mins = test_results["min"]
    test_median = test_results["median"]
    
    # 3. Predict (Using the Mean score vs Threshold)
    predictions = (test_means < prob_threshold).astype(int)
    
    # 4. Metrics
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=["Benign", "Attack"]))
    
    cm = confusion_matrix(test_labels, predictions)
    print(f"Confusion Matrix:\nTP: {cm[1][1]} | FN: {cm[1][0]}\nFP: {cm[0][1]} | TN: {cm[0][0]}")

    print(f"ROC AUC (on Mean): {roc_auc_score(test_labels, -test_means):.4f}")

    # Save Results CSV with MIN, MEAN, MAX
    results_df = pd.DataFrame({
        'Label_True': test_labels,
        'Label_Pred': predictions,
        'Score_Mean': test_means,
        'Score_Min': test_mins,
        'Score_Median': test_median
    })
    
    output_csv_path = "/home/spritz/storage/disk0/Master_Thesis/Stuff/detection_results_sliding_window.csv"
    results_df.to_csv(output_csv_path, index=False)
    print(f"Done. Results saved to {output_csv_path}")