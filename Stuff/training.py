import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    T5ForConditionalGeneration, 
    Trainer, 
    TrainingArguments
)

# --- 1. Updated Data Loading with Context ---
def load_context_data(filepath):
    """
    Reads Payload (col 0), Source (col 3), and Dest (col 4).
    Combines them into: "S:<src> D:<dst> P:<payload_bytes>"
    """
    # Read columns: 0=Payload, 3=Src, 4=Dst
    # We assume the file structure is: payload, cat, type, src, dst, time
    df = pd.read_csv(filepath, header=None, usecols=[0, 3, 4], names=['payload', 'src', 'dst'])
    
    processed_texts = []
    
    # Iterate through rows to format the data
    for _, row in df.iterrows():
        hex_str = str(row['payload'])
        
        # 1. Convert Payload Hex -> Latin-1 Bytes
        try:
            payload_bytes = bytes.fromhex(hex_str).decode('latin-1')
        except:
            continue # Skip corrupt lines
            
        # 2. Get Source and Dest as strings
        src = str(row['src'])
        dst = str(row['dst'])
        
        # 3. Combine into Context String
        # "S:3 D:1 P:" followed by the raw byte characters
        full_text = f"S:{src} D:{dst} P:{payload_bytes}"
        
        processed_texts.append(full_text)
        
    print(f"Loaded {len(processed_texts)} context-aware packets from {filepath}")
    return processed_texts

# --- 2. Dataset Class (Unchanged) ---
# The Dataset class works fine because it just tokenizes whatever string we give it.
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
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# --- 3. Main Training Loop ---
if __name__ == "__main__":
    # A. Settings
    MODEL_NAME = "google/byt5-small"
    MAX_LENGTH = 270  # Increased slightly to account for "S:XX D:XX P:" prefix
    BATCH_SIZE = 8
    EPOCHS = 3  # You might need 5 epochs now as the task is slightly harder
    
    # B. Load Data (Using the NEW function)
    print("Loading Training Data...")
    train_texts = load_context_data("/home/spritz/storage/disk0/Master_Thesis/Dataset/splits/train.txt")
    print("Loading Validation Data...")
    val_texts = load_context_data("/home/spritz/storage/disk0/Master_Thesis/Dataset/splits/validation.txt")

    # C. Initialize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # D. Create Datasets
    train_dataset = ModbusDataset(train_texts, tokenizer, max_length=MAX_LENGTH)
    val_dataset = ModbusDataset(val_texts, tokenizer, max_length=MAX_LENGTH)

    # E. Arguments (Using Gradient Accumulation for Memory Safety)
    training_args = TrainingArguments(
        output_dir="./modbus_context_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=EPOCHS,
        save_total_limit=2,
        logging_steps=100,
        fp16=False, 
        report_to="none"
    )

    # F. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting Context-Aware Training...")
    trainer.train()
    
    print("Saving Model...")
    trainer.save_model("./final_modbus_context_autoencoder")
    tokenizer.save_pretrained("./final_modbus_context_autoencoder")
    print("Done!")