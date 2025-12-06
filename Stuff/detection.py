import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score

# --- 1. Setup & Paths ---
MODEL_PATH = "/home/spritz/storage/disk0/Master_Thesis/Stuff/Byt5/simplified-modbus-reconstruction-finetuned"
VAL_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset/simplified_dataset/simple_mixed_val.csv"
TEST_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset/simplified_dataset/simple_mixed_test.csv"
SAVE_RESULTS_PATH = "/home/spritz/storage/disk0/Master_Thesis/Stuff/detection_results_prob.csv" 
BATCH_SIZE = 32
MAX_LENGTH = 64
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

# --- 3. Dataset Class ---
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
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, 
            max_length=self.max_length,
            padding="max_length", 
            truncation=True, 
            return_tensors="pt")
        
        original_input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        
        # Labels = Original Clean Input
        labels = original_input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Input = Masked Input
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

# --- 4. The Core: Calculate Probabilities (FIXED) ---
def get_metrics_and_reconstruction(model, dataset, device, tokenizer, return_reconstruction=False):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    
    all_losses = []
    all_mean_probs = []
    all_min_probs = []
    all_reconstructions = []

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none') 

    print(f"Processing {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device) 
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels)
            
            logits = outputs.logits 

            # Shift for Next-Token Prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # --- 1. Loss Calculation ---
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())
            
            active_mask = shift_labels != -100
            valid_tokens_count = active_mask.sum(dim=1).clamp(min=1)
            
            sample_losses = (loss * active_mask).sum(dim=1) / valid_tokens_count
            all_losses.extend(sample_losses.cpu().numpy())

            # --- 2. Probability Calculation ---
            probs = torch.softmax(shift_logits, dim=-1)

            # FIX: Replace -100 with 0 temporarily for gather
            gather_indices = shift_labels.clone()
            gather_indices[gather_indices == -100] = 0

            # Get prob of TRUE label
            true_token_probs = probs.gather(2, gather_indices.unsqueeze(-1)).squeeze(-1)

            # Mean Prob
            sample_mean_probs = (true_token_probs * active_mask).sum(dim=1) / valid_tokens_count
            all_mean_probs.extend(sample_mean_probs.cpu().numpy())

            # Min Prob
            masked_probs_for_min = true_token_probs.clone()
            masked_probs_for_min[~active_mask] = 1.0 
            sample_min_probs, _ = masked_probs_for_min.min(dim=1)
            all_min_probs.extend(sample_min_probs.cpu().numpy())

            # --- 3. Reconstruction ---
            if return_reconstruction:
                predicted_ids = torch.argmax(logits, dim=-1)
                decoded_batch = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
                all_reconstructions.extend(decoded_batch)
            
    if return_reconstruction:
        return np.array(all_losses), np.array(all_mean_probs), np.array(all_min_probs), all_reconstructions
    else:
        return np.array(all_losses), np.array(all_mean_probs), np.array(all_min_probs)

# --- 5. Main Execution ---
if __name__ == "__main__":
    print("Starting Detection...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # A. Load Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True).to(device)

    # B. Load Data
    print("Loading Data...")
    val_texts, val_labels = load_labeled_context_data(VAL_PATH)
    test_texts, test_labels = load_labeled_context_data(TEST_PATH)

    val_dataset = ModbusDataset(val_texts, tokenizer, max_length=MAX_LENGTH, mask_prob=MASK_PROBABILITY)
    test_dataset = ModbusDataset(test_texts, tokenizer, max_length=MAX_LENGTH, mask_prob=MASK_PROBABILITY)

    # C. Calculate Metrics
    print("Calculating Validation Metrics...")
    val_losses, val_mean_probs, val_min_probs = get_metrics_and_reconstruction(
        model, val_dataset, device, tokenizer, return_reconstruction=False)
    
    print("Calculating Test Metrics...")
    test_losses, test_mean_probs, test_min_probs, test_reconstructions = get_metrics_and_reconstruction(
        model, test_dataset, device, tokenizer, return_reconstruction=True)

    # D. Determine Threshold
    # We use the 1st percentile of Validation Mean Probabilities as the cutoff
    # Anything with LOWER probability than this is considered anomalous
    prob_threshold = np.percentile(val_mean_probs, 1) 
    print(f"\n--- Results ---")
    print(f"Probability Threshold (1st percentile Val): {prob_threshold:.6f}")

    # E. Predict
    # Normal: Prob > Threshold
    # Attack: Prob < Threshold
    predictions = (test_mean_probs < prob_threshold).astype(int)
    
    # F. Confusion Matrix & Metrics
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=["Benign", "Attack"]))
    
    # --- ADDED BACK CONFUSION MATRIX ---
    cm = confusion_matrix(test_labels, predictions)
    print("\nConfusion Matrix:")
    print(f"True Benign:        {cm[0][0]} | False Attack (FP): {cm[0][1]}")
    print(f"Missed Attack (FN): {cm[1][0]} | True Attack (TP):  {cm[1][1]}")
    # -----------------------------------

    auc = roc_auc_score(test_labels, 1 - test_mean_probs) # Invert because low prob = high anomaly score
    print(f"ROC AUC Score: {auc:.4f}")

    benign_probs = test_mean_probs[np.array(test_labels) == 0]
    attack_probs = test_mean_probs[np.array(test_labels) == 1]

    print(f"\n--- Statistics ---")
    print(f"Benign -> Mean Confidence: {np.mean(benign_probs):.6f}")
    print(f"Attack -> Mean Confidence: {np.mean(attack_probs):.6f}")

    # G. Save
    print(f"\nSaving results to {SAVE_RESULTS_PATH}...")
    results_df = pd.DataFrame({
        'Original': test_texts,
        'Reconstructed': test_reconstructions,
        'Label_True': test_labels,
        'Label_Pred': predictions,
        'Loss_Score': test_losses,
        'Mean_Prob_Score': test_mean_probs,
        'Min_Prob_Score': test_min_probs
    })
    results_df.to_csv(SAVE_RESULTS_PATH, index=False)
    print("Done.")