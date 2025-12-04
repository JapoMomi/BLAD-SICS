import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score

# --- 1. Setup & Paths ---
MODEL_PATH = "/home/spritz/storage/disk0/Master_Thesis/Stuff/Byt5/final_simplied_context_model"
VAL_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset/simplified_dataset/simple_mixed_val.csv"
TEST_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset/simplified_dataset/simple_mixed_test.csv"
SAVE_RESULTS_PATH = "detection_results.csv" # <--- New file output
BATCH_SIZE = 32
MAX_LENGTH = 64

# --- 2. Data Loading ---
def load_labeled_context_data(filepath):
    """
    Reads Payload, Label.
    """
    # Force dtype=str to prevent hex/int confusion
    df = pd.read_csv(
        filepath, 
        header=None, 
        usecols=[0, 1, 2], 
        names=['payload', 'cat', 'type'],
        dtype={'payload': str}, 
        keep_default_na=False
    )
    
    # Generate Labels
    df['is_attack'] = ((df['cat'] != 0) | (df['type'] != 0)).astype(int)
    
    processed_texts = []
    final_labels = []
    
    for _, row in df.iterrows():
        text_content = str(row['payload']).strip()
        
        # Safety check
        if len(text_content) > 0:
            processed_texts.append(text_content)
            final_labels.append(row['is_attack'])
        
    return processed_texts, final_labels

# --- 3. Dataset Class ---
class ModbusDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
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
            padding="max_length", 
            truncation=True, 
            return_tensors="pt")
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# --- 4. The Core: Calculate Reconstruction & Capture Output ---
def get_losses_and_reconstruction(model, dataset, device, tokenizer, return_reconstruction=False):
    """
    Returns:
        - losses (numpy array)
        - reconstructed_texts (list of strings, only if return_reconstruction=True)
    """
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    
    all_losses = []
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

            # --- A. Calculate Loss ---
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())

            active_mask = shift_labels != -100
            sample_losses = (loss * active_mask).sum(dim=1) / active_mask.sum(dim=1)
            all_losses.extend(sample_losses.cpu().numpy())

            # --- B. Decode Reconstruction (Optional) ---
            if return_reconstruction:
                # Argmax gets the most likely token ID at each step
                # We use the full logits (not shifted) because we want to see the model's full output
                predicted_ids = torch.argmax(logits, dim=-1)
                
                # Decode back to string
                decoded_batch = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
                all_reconstructions.extend(decoded_batch)
            
    if return_reconstruction:
        return np.array(all_losses), all_reconstructions
    else:
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
    print("Loading Validation Data...")
    val_texts, val_labels = load_labeled_context_data(VAL_PATH)
    
    print("Loading Test Data...")
    test_texts, test_labels = load_labeled_context_data(TEST_PATH)

    # C. Prepare Datasets
    val_dataset = ModbusDataset(val_texts, tokenizer, max_length=MAX_LENGTH)
    test_dataset = ModbusDataset(test_texts, tokenizer, max_length=MAX_LENGTH)

    # D. Calculate Errors
    # 1. Validation (We don't need reconstructions here, just losses for threshold)
    val_losses = get_losses_and_reconstruction(model, val_dataset, device, tokenizer, return_reconstruction=False)
    
    # 2. Test (We WANT reconstructions here to save to file)
    test_losses, test_reconstructions = get_losses_and_reconstruction(model, test_dataset, device, tokenizer, return_reconstruction=True)

    # E. Determine Threshold
    threshold = np.percentile(val_losses, 99) 
    print(f"\n--- Results ---")
    print(f"Calculated Threshold (99th percentile of Val): {threshold:.6f}")

    # F. Predict
    # If Error > Threshold -> Predict 1 (Attack)
    predictions = (test_losses > threshold).astype(int)
    
    # G. Metrics
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=["Benign", "Attack"]))
    
    auc = roc_auc_score(test_labels, test_losses)
    f1 = f1_score(test_labels, predictions)
    
    print(f"ROC AUC Score: {auc:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    
    cm = confusion_matrix(test_labels, predictions)
    print("\nConfusion Matrix:")
    print(f"True Benign:  {cm[0][0]} | False Attack: {cm[0][1]} (False Positives)")
    print(f"Missed Attack:{cm[1][0]} | True Attack:  {cm[1][1]} (True Positives)")

    benign_losses = test_losses[np.array(test_labels) == 0]
    attack_losses = test_losses[np.array(test_labels) == 1]

    print(f"\n--- Statistics ---")
    print(f"Benign Traffic -> Mean Loss: {np.mean(benign_losses):.6f}")
    print(f"Attack Traffic -> Mean Loss: {np.mean(attack_losses):.6f}")

    # --- H. SAVE RESULTS TO CSV ---
    print(f"\nSaving detailed results to {SAVE_RESULTS_PATH}...")
    
    # Create DataFrame
    results_df = pd.DataFrame({
        'Original_Packet': test_texts,
        'Reconstructed_Packet': test_reconstructions,
        'Label_True': test_labels,
        'Label_Pred': predictions,
        'Loss_Score': test_losses,
        'Is_Correct': (np.array(test_labels) == predictions) # Helper column to sort by errors
    })

    # Save
    results_df.to_csv(SAVE_RESULTS_PATH, index=False)
    print("File saved successfully.")
    
    # Print a few examples of Misclassifications to the console for quick check
    print("\n--- Examples of Misclassified Packets (if any) ---")
    errors = results_df[results_df['Is_Correct'] == False].head(5)
    if not errors.empty:
        for idx, row in errors.iterrows():
            print(f"Original: {row['Original_Packet']}")
            print(f"Reconst:  {row['Reconstructed_Packet']}")
            print(f"Loss: {row['Loss_Score']:.4f} (Threshold: {threshold:.4f})")
            print("-" * 30)
    else:
        print("No errors found! Perfect classification.")