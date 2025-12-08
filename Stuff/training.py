import random
import datasets
import numpy as np
import pandas as pd
import torch
from transformers import (
    ByT5Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)

# --- CONFIGURATION ---
model_checkpoint = "google/byt5-small"  
train_file = "/home/spritz/storage/disk0/Master_Thesis/Dataset/simplified_dataset/simple_mixed_train.csv" 
valid_file = "/home/spritz/storage/disk0/Master_Thesis/Dataset/simplified_dataset/simple_mixed_val.csv" 
MASK_PROBABILITY = 0.25 

COLUMN_NAMES = ["packet", "type_1", "type_2", "src", "dst", "timestamp"]

def filter_normal_packets(data_row):
    if data_row["packet"] is None or data_row["packet"] == "":
        return False
    # Keep only normal packets (0,0)
    if data_row["type_1"] == 0 and data_row["type_2"] == 0:
        return True
    return False

def hex_to_bytes(example):
    """Converts Hex String packet to Raw Bytes String"""
    try:
        # Convert hex string to bytes, then decode to utf-8 (ignoring errors to keep it as string)
        # ByT5 tokenizer will handle the raw bytes underneath
        raw_content = bytes.fromhex(example["packet"]).decode('utf-8', errors='ignore')
        example["packet"] = raw_content
    except Exception:
        # Keep original if fail
        pass
    return example

def convert_to_features(example_batch):
    # 1. Tokenize inputs
    model_inputs = tokenizer(
        example_batch["packet"], 
        truncation=True, 
        padding=False, 
        max_length=128
    )
    
    # 2. Tokenize targets (Copy of input)
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            example_batch["packet"], 
            truncation=True, 
            padding=False, 
            max_length=128
        )
    
    # 3. Apply MASKING
    mask_token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    batch_input_ids = model_inputs["input_ids"]
    masked_batch = []
    
    for input_ids in batch_input_ids:
        arr = np.array(input_ids)
        probability_matrix = np.random.rand(*arr.shape)
        masked_indices = probability_matrix < MASK_PROBABILITY
        arr[masked_indices] = mask_token_id
        masked_batch.append(arr.tolist())
        
    model_inputs["input_ids"] = masked_batch
    model_inputs["labels"] = target_encodings["input_ids"]
    return model_inputs

# Initialize
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
tokenizer = ByT5Tokenizer.from_pretrained(model_checkpoint)

dataset = datasets.load_dataset(
    "csv", 
    sep=",", 
    names=COLUMN_NAMES, 
    data_files={"train": [train_file], "valid": [valid_file]}
)

# Filter
print(f"Original train size: {len(dataset['train'])}")
dataset = dataset.filter(filter_normal_packets)
print(f"Filtered size: {len(dataset['train'])}")

# CONVERT HEX TO BYTES BEFORE TOKENIZATION
print("Converting Hex to Bytes...")
dataset = dataset.map(hex_to_bytes)

# Tokenize
dataset = dataset.map(
    convert_to_features, 
    batched=True, 
    num_proc=4, 
    remove_columns=COLUMN_NAMES 
)

# Training Args
finetuned_model_name = "/home/spritz/storage/disk0/Master_Thesis/Stuff/Byt5/simplified-hex_modbus-reconstruction-finetuned"
args = Seq2SeqTrainingArguments(
    output_dir=finetuned_model_name,
    overwrite_output_dir=True,
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    num_train_epochs=10,
    predict_with_generate=True,
    generation_max_length=128,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
    report_to="none",
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained(finetuned_model_name)
tokenizer.save_pretrained(finetuned_model_name)