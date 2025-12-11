import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# --- 1. Setup ---
MODEL_PATH = "/home/spritz/storage/disk0/Master_Thesis/Stuff/Byt5/simplified-hex_modbus-sequence_5-finetuned" # Path updated to match training output
VAL_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset/simplified_dataset/simple_mixed_val.csv"
TEST_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset/simplified_dataset/simple_mixed_test.csv"
SAVE_RESULTS_PATH = "/home/spritz/storage/disk0/Master_Thesis/Stuff/detection_results_sequences.csv" # Updated name
BATCH_SIZE = 32
MASK_PROBABILITY = 0.25 

# --- NEW SEQUENCE CONFIGURATION ---
SEQUENCE_LENGTH = 5        # N packets per sequence
MAX_LENGTH = 1024          # New sequence max length
PACKET_SEPARATOR = '     ' # 5 blank spaces
# ----------------------------------

# --- 2. Data Loading (Modified for Sequences) ---
def load_labeled_context_data(filepath, N, separator):
    df = pd.read_csv(
        filepath, 
        header=None, 
        usecols=[0, 1, 2], 
        names=['payload', 'cat', 'type'],
        dtype={'payload': str}, 
        keep_default_na=False
    )
    
    # Label each individual packet
    is_attack_labels = ((df['cat'] != 0) | (df['type'] != 0)).astype(int).tolist()
    
    processed_sequences = []
    final_sequence_labels = []
    
    # Group into sequences of N
    for i in range(0, len(df), N):
        chunk = df.iloc[i:i + N]
        if len(chunk) < N: # Drop incomplete last chunk
            continue

        # Concatenate packet payloads
        sequence_payload = separator.join(chunk['payload'].tolist())
        
        # Sequence label: 1 if max label in chunk is 1, 0 otherwise
        chunk_labels = is_attack_labels[i:i+N]
        sequence_label = max(chunk_labels)
        
        processed_sequences.append(sequence_payload)
        final_sequence_labels.append(sequence_label)

    return processed_sequences, final_sequence_labels

# --- 3. Dataset Class (Re-using logic, max_length is the only change) ---
class ModbusDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, mask_prob):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # hex_text is now the sequence of concatenated hex strings
        hex_text = self.texts[idx]
        
        # --- CONVERT HEX SEQUENCE STRING TO RAW BYTES SEQUENCE STRING ---
        packet_list = hex_text.split(PACKET_SEPARATOR)
        raw_content_list = []
        for hex_segment in packet_list:
            # Decode each segment individually to handle spaces gracefully
            try:
                 raw_content_list.append(bytes.fromhex(hex_segment).decode('utf-8', errors='ignore'))
            except:
                 raw_content_list.append(hex_segment) # Fallback to hex if decoding fails
                 
        raw_content = PACKET_SEPARATOR.join(raw_content_list)
        
        # Encoding using the new MAX_LENGTH
        encoding = self.tokenizer(
            raw_content, 
            max_length=self.max_length,
            padding="max_length", 
            truncation=True, 
            return_tensors="pt")
        
        original_input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        
        labels = original_input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        input_ids = original_input_ids.clone()
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices = masked_indices & (input_ids != self.tokenizer.pad_token_id)
        input_ids[masked_indices] = self.mask_token_id

        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "labels": labels
        }

# --- 4. The Core: Calculate Log-Probabilities (Unchanged) ---
def get_logprob_metrics(model, dataset, device, tokenizer, return_reconstruction=False):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    
    all_mean_logprobs = []
    all_min_logprobs = []
    all_reconstructions = []

    print(f"Processing {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device) 
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits 

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            active_mask = shift_labels != -100
            valid_tokens_count = active_mask.sum(dim=1).clamp(min=1)

            # Use Log Softmax to avoid underflow
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

            gather_indices = shift_labels.clone()
            gather_indices[gather_indices == -100] = 0

            true_token_log_probs = log_probs.gather(2, gather_indices.unsqueeze(-1)).squeeze(-1)

            # A. Mean Log-Prob
            sample_mean_logprobs = (true_token_log_probs * active_mask).sum(dim=1) / valid_tokens_count
            all_mean_logprobs.extend(sample_mean_logprobs.cpu().numpy())

            # B. Min Log-Prob
            # Set ignored tokens to 0.0 (max probability) so they don't affect min
            masked_log_probs = true_token_log_probs.clone()
            masked_log_probs[~active_mask] = 0.0 
            sample_min_logprobs, _ = masked_log_probs.min(dim=1)
            all_min_logprobs.extend(sample_min_logprobs.cpu().numpy())

            # C. Reconstruction
            if return_reconstruction:
                predicted_ids = torch.argmax(logits, dim=-1)
                decoded_batch = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
                # Trim to original length
                true_lengths = (labels != -100).sum(dim=1) - 1
                clean_batch = []
                for text, length in zip(decoded_batch, true_lengths):
                    # Need to be careful here: the length is in BYTES, not characters.
                    # We will simply return the decoded batch without complex trimming.
                    clean_batch.append(text) 
                all_reconstructions.extend(clean_batch)
            
    # Ensure consistent returns
    if return_reconstruction:
        return np.array(all_mean_logprobs), np.array(all_min_logprobs), all_reconstructions
    else:
        return np.array(all_mean_logprobs), np.array(all_min_logprobs)

# --- 5. Main Execution ---
if __name__ == "__main__":
    print(f"Starting Sequence Detection (N={SEQUENCE_LENGTH} with '{PACKET_SEPARATOR}' separator)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True).to(device)

    # Load data as sequences
    val_texts, val_labels = load_labeled_context_data(VAL_PATH, SEQUENCE_LENGTH, PACKET_SEPARATOR)
    test_texts, test_labels = load_labeled_context_data(TEST_PATH, SEQUENCE_LENGTH, PACKET_SEPARATOR)

    val_dataset = ModbusDataset(val_texts, tokenizer, max_length=MAX_LENGTH, mask_prob=MASK_PROBABILITY)
    test_dataset = ModbusDataset(test_texts, tokenizer, max_length=MAX_LENGTH, mask_prob=MASK_PROBABILITY)

    # Validation
    val_mean_log, val_min_log = get_logprob_metrics(
        model, val_dataset, device, tokenizer, return_reconstruction=False)
    
    # Test (Calculates reconstructions too)
    test_mean_log, test_min_log, test_reconstructions = get_logprob_metrics(
        model, test_dataset, device, tokenizer, return_reconstruction=True)

    # Threshold (Using 1st percentile of validation data)
    prob_threshold = np.percentile(val_min_log, 1) 
    print(f"\nLog-Prob Threshold (1st percentile): {prob_threshold:.6f}")

    # Predict (Lower score = Attack)
    predictions = (test_min_log < prob_threshold).astype(int)
    
    # Metrics
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=["Benign", "Attack"]))
    
    cm = confusion_matrix(test_labels, predictions)
    print(f"Confusion Matrix:\nTP: {cm[1][1]} | FN: {cm[1][0]}\nFP: {cm[0][1]} | TN: {cm[0][0]}")

    print(f"ROC AUC: {roc_auc_score(test_labels, -test_min_log):.4f}")

    # Save
    results_df = pd.DataFrame({
        'Original_Sequence_Hex': test_texts,
        'Reconstructed_Sequence_Bytes': test_reconstructions,
        'Label_True': test_labels,
        'Label_Pred': predictions,
        'Mean_LogProb': test_mean_log,
        'Min_LogProb': test_min_log
    })
    results_df.to_csv(SAVE_RESULTS_PATH, index=False)
    print("Done.")