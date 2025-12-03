import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    T5ForConditionalGeneration, 
    Trainer, 
    TrainingArguments
)

def load_context_data(filepath):
    """
    Reads Payload (col 0), Source (col 3), and Dest (col 4).
    Combines them into: "S:<src> D:<dst> P:<payload_bytes>"
    """
    df = pd.read_csv(filepath, header=None, usecols=[0], names=['payload'])
    
    processed_texts = []
    
    # Iterate through rows to format the data
    for _, row in df.iterrows():
        hex_str = str(row['payload'])
        
        # 1. Convert Payload Hex -> Latin-1 Bytes
        # e.g. \x04\x03
        try:
            payload_bytes = bytes.fromhex(hex_str).decode('latin-1')
        except:
            continue # Skip corrupt lines
        
        # 3. Combine into Context String
        full_text = f"{payload_bytes}"
        
        processed_texts.append(full_text)
        
    print(f"Loaded {len(processed_texts)} context-aware packets from {filepath}")
    return processed_texts

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
        #The tokenizer usually outputs a shape of [1, 128] (Batch Size, Sequence Length). 
        #But the DataLoader adds the batch dimension later. 
        # If we don't squeeze (remove the 1), we end up with [Batch, 1, 128], which causes shape mismatch errors. We want just [128]
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        
        labels = input_ids.clone()
        #Not want the model to learn how to predict padding
        #In PyTorch/Hugging Face, setting a label to -100 is a special code that means "Ignore this number when calculating loss (error)."
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

if __name__ == "__main__":
    # A. Settings
    MODEL_NAME = "google/byt5-small"
    MAX_LENGTH = 256  # Increased slightly to account for "S:XX D:XX P:" prefix
    BATCH_SIZE = 16
    EPOCHS = 3  # You might need 5 epochs now as the task is slightly harder
    
    # B. Load Data (Using the NEW function)
    print("Loading Training Data...")
    train_texts = load_context_data("/home/spritz/storage/disk0/Master_Thesis/Dataset/simplified_dataset/simple_mixed_train.csv")
    print("Loading Validation Data...")
    val_texts = load_context_data("/home/spritz/storage/disk0/Master_Thesis/Dataset/simplified_dataset/simple_mixed_val.csv")

    # C. Initialize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, local_files_only=True)

    # D. Create Datasets
    train_dataset = ModbusDataset(train_texts, tokenizer, max_length=MAX_LENGTH)
    val_dataset = ModbusDataset(val_texts, tokenizer, max_length=MAX_LENGTH)

    # E. Arguments (Using Gradient Accumulation for Memory Safety)
    training_args = TrainingArguments(
        output_dir="/home/spritz/storage/disk0/Master_Thesis/Stuff/Byt5/simplied_context_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
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
    trainer.save_model("/home/spritz/storage/disk0/Master_Thesis/Stuff/Byt5/final_simplied_context_model")
    tokenizer.save_pretrained("/home/spritz/storage/disk0/Master_Thesis/Stuff/Byt5/final_simplied_context_model")
    print("Done!")