import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# --- 1. Setup ---
MODEL_PATH = "/home/spritz/storage/disk0/Master_Thesis/Stuff/Byt5/simplified-hex_modbus-reconstruction-finetuned"
VAL_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset/simplified_dataset/simple_mixed_val.csv"
TEST_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset/simplified_dataset/simple_mixed_test.csv"
SAVE_RESULTS_PATH = "/home/spritz/storage/disk0/Master_Thesis/Stuff/detection_results_prob_only.csv" 
BATCH_SIZE = 32
MAX_LENGTH = 128 # Increased for safety
MASK_PROBABILITY = 0.15 

# --- 2. Data Loading ---
def load_labeled_context_data(filepath):
    df = pd.read_csv(
        filepath, 
        header=None, 
        usecols=[0, 1, 2], 
        names=['payload', 'cat', 'type'],
        dtype={'payload': str}, 
        keep_default_na=False
    )
    df['is_attack'] = ((df['cat'] != 0) | (df['type'] != 0)).astype(int)
    
    processed_texts = []
    final_labels = []
    
    for _, row in df.iterrows():
        text_content = str(row['payload']).strip()
        if len(text_content) > 0:
            processed_texts.append(text_content)
            final_labels.append(row['is_attack'])
        
    return processed_texts, final_labels

# --- 3. Dataset Class (With Hex Conversion) ---
class ModbusDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128, mask_prob=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        hex_text = self.texts[idx]
        
        # --- CONVERT HEX STRING TO RAW BYTES ---
        try:
            # Decode hex to raw bytes string for ByT5
            raw_content = bytes.fromhex(hex_text).decode('utf-8', errors='ignore')
        except:
            raw_content = hex_text # Fallback
            
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

# --- 4. The Core: Calculate Log-Probabilities ---
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
                    clean_batch.append(text[:max(0, int(length.item()))])
                all_reconstructions.extend(clean_batch)
            
    # Ensure consistent returns
    if return_reconstruction:
        return np.array(all_mean_logprobs), np.array(all_min_logprobs), all_reconstructions
    else:
        return np.array(all_mean_logprobs), np.array(all_min_logprobs)

# --- 5. Main Execution ---
if __name__ == "__main__":
    print("Starting Detection (Hex->Bytes + Log-Prob)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True).to(device)

    val_texts, val_labels = load_labeled_context_data(VAL_PATH)
    test_texts, test_labels = load_labeled_context_data(TEST_PATH)

    val_dataset = ModbusDataset(val_texts, tokenizer, max_length=MAX_LENGTH, mask_prob=MASK_PROBABILITY)
    test_dataset = ModbusDataset(test_texts, tokenizer, max_length=MAX_LENGTH, mask_prob=MASK_PROBABILITY)

    # Validation
    val_mean_log, val_min_log = get_logprob_metrics(
        model, val_dataset, device, tokenizer, return_reconstruction=False)
    
    # Test (Calculates reconstructions too)
    test_mean_log, test_min_log, test_reconstructions = get_logprob_metrics(
        model, test_dataset, device, tokenizer, return_reconstruction=True)

    # Threshold
    prob_threshold = np.percentile(val_min_log, 1) 
    print(f"\nLog-Prob Threshold: {prob_threshold:.6f}")

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
        'Original_Hex': test_texts,
        'Reconstructed_Bytes': test_reconstructions,
        'Label_True': test_labels,
        'Label_Pred': predictions,
        'Mean_LogProb': test_mean_log,
        'Min_LogProb': test_min_log
    })
    results_df.to_csv(SAVE_RESULTS_PATH, index=False)
    print("Done.")