print("1. Importing pandas...")
import pandas as pd
print("2. Importing torch...")
import torch
from torch.utils.data import DataLoader, Dataset
print("3. Importing numpy...")
import numpy as np
print("4. Importing transformers...")
from transformers import AutoTokenizer, T5ForConditionalGeneration
print("5. Importing sklearn...")
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score
print("DONE. Imports are fine.") 

# --- 1. Setup & Paths ---
# Use the path where you just saved the model
MODEL_PATH = "/home/spritz/storage/disk0/Master_Thesis/Stuff/Byt5/final_modbus_autoencoder"
VAL_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset/splits/validation.txt"
TEST_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset/splits/test.txt"
BATCH_SIZE = 32
MAX_LENGTH = 256

# --- 2. Data Loading (Updated for Labels) ---
def load_labeled_context_data(filepath):
    """
    Reads Payload, Src, Dst, Labels.
    Returns: formatted strings list, labels list
    """
    # Read cols: 0=payload, 1=cat, 2=type, 3=src, 4=dst
    df = pd.read_csv(filepath, header=None, usecols=[0, 1, 2, 3, 4], names=['payload', 'cat', 'type', 'src', 'dst'])
    
    # Generate Labels (Same as before)
    df['is_attack'] = ((df['cat'] != 0) | (df['type'] != 0)).astype(int)
    
    processed_texts = []
    final_labels = []
    
    for _, row in df.iterrows():
        hex_str = str(row['payload'])
        try:
            payload_bytes = bytes.fromhex(hex_str).decode('latin-1')
        except:
            continue
            
        src = str(row['src'])
        dst = str(row['dst'])
        
        # Combine
        full_text = f"S:{src} D:{dst} P:{payload_bytes}"
        
        processed_texts.append(full_text)
        final_labels.append(row['is_attack'])
        
    return processed_texts, final_labels

# --- 3. Dataset Class (Same as before) ---
class ModbusDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, 
            max_length=self.max_length,
            padding="max_length",  # Adds the [PAD] tokens to reach max_length
            truncation=True, 
            return_tensors="pt")
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# --- 4. The Core: Calculate Reconstruction Error ---
def get_losses(model, dataset, device):
    """
    Runs inference and returns the CrossEntropyLoss for EACH sample.
    """
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    all_losses = []

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none') # 'none' gives loss per token

    print(f"Calculating reconstruction errors for {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            # binary "filter" that tells the model which parts of the input are real data and which parts are just empty padding
            attention_mask = batch['attention_mask'].to(device) 
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels)
            # (raw prediction scores) -> Logits: predict the NEXT token
            logits = outputs.logits

            # To correctly calculate the error, we shift the logits one step forward and
            # the labels one step back so that the prediction for token t is compared against the true token t+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Calculate loss per token
            # View(-1) flattens the batch to 1D list for calculation
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Reshape back to [batch, seq_len]
            loss = loss.view(shift_labels.size())

            # Average loss per sample (ignoring padding -100)
            active_mask = shift_labels != -100
            # Sum of errors / Number of non-padding tokens
            sample_losses = (loss * active_mask).sum(dim=1) / active_mask.sum(dim=1)
            
            all_losses.extend(sample_losses.cpu().numpy())
            
    return np.array(all_losses)

# --- 5. Main Execution ---
if __name__ == "__main__":
    print("Starting")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # A. Load Model
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True).to(device)

    # B. Load Data
    print("Loading Validation Data (For Threshold)...")
    val_texts, val_labels = load_labeled_context_data(VAL_PATH)
    
    print("Loading Test Data (For Evaluation)...")
    test_texts, test_labels = load_labeled_context_data(TEST_PATH)

    # C. Prepare Datasets
    val_dataset = ModbusDataset(val_texts, tokenizer, max_length=MAX_LENGTH)
    test_dataset = ModbusDataset(test_texts, tokenizer, max_length=MAX_LENGTH)

    # D. Calculate Errors
    # 1. Get errors on Benign Validation set to establish "Normal"
    val_losses = get_losses(model, val_dataset, device)
    
    # 2. Get errors on Test set
    test_losses = get_losses(model, test_dataset, device)

    # E. Determine Threshold
    # Strategy: Threshold is the value where 99% of validation data is included.
    # Anything higher than this is an anomaly.
    threshold = np.percentile(val_losses, 99) 
    print(f"\n--- Results ---")
    print(f"Calculated Threshold (99th percentile of Val): {threshold:.6f}")

    # F. Predict & Evaluate
    # If Error > Threshold -> Predict 1 (Attack)
    predictions = (test_losses > threshold).astype(int)
    
    # G. Metrics
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=["Benign", "Attack"]))
    
    # Calculate specific scores
    auc = roc_auc_score(test_labels, test_losses)
    f1 = f1_score(test_labels, predictions) # Default is binary (Attack class)
    
    print(f"ROC AUC Score: {auc:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    
    # Optional: Confusion Matrix for a quick look at False Positives vs False Negatives
    cm = confusion_matrix(test_labels, predictions)
    print("\nConfusion Matrix:")
    print(f"True Benign:  {cm[0][0]} | False Attack: {cm[0][1]} (False Positives)")
    print(f"Missed Attack:{cm[1][0]} | True Attack:  {cm[1][1]} (True Positives)")

    # Separate losses
    benign_losses = test_losses[np.array(test_labels) == 0]
    attack_losses = test_losses[np.array(test_labels) == 1]

    print(f"\n--- Statistics ---")
    print(f"Benign Traffic -> Mean Loss: {np.mean(benign_losses):.6f} | Std Dev: {np.std(benign_losses):.6f}")
    print(f"Attack Traffic -> Mean Loss: {np.mean(attack_losses):.6f} | Std Dev: {np.std(attack_losses):.6f}")
    print(f"\nCurrent Threshold: {threshold:.6f}")

    print(f"\n--- Max/Min Values ---")
    print(f"Max Benign Loss: {np.max(benign_losses):.6f}")
    print(f"Min Attack Loss: {np.min(attack_losses):.6f}")