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

# Hyperparameters
MASK_PROBABILITY = 0.25
SEQUENCE_LENGTH = 5  # Number of packets to reconstruct as a sequence
SEPARATOR = "     "  # 5 blank spaces
MAX_TOKEN_LENGTH = 512 # Increased to accommodate N packets + separators

COLUMN_NAMES = ["packet", "type_1", "type_2", "src", "dst", "timestamp"]

def hex_to_bytes(example):
    """Converts Hex String packet to Raw Bytes String"""
    try:
        # Convert hex string to bytes, then decode to utf-8 (ignoring errors to keep it as string)
        if example["packet"]:
            raw_content = bytes.fromhex(example["packet"]).decode('utf-8', errors='ignore')
            example["packet"] = raw_content
        else:
            example["packet"] = ""
    except Exception:
        pass # Keep original if fail
    return example

def group_into_sequences(examples):
    """
    Groups packets into sequences of length N.
    Logic: 
    - Concatenates N packets with 5 spaces separator.
    - Sequence is 'Normal' (0) if ALL packets are normal.
    - Sequence is 'Attack' (1) if ANY packet is an attack.
    """
    # Extract lists from the batch
    packets = examples["packet"]
    type1 = examples["type_1"]
    type2 = examples["type_2"]
    
    grouped_packets = []
    # We will just use type_1 for the sequence label for simplicity, 
    # but check both types to determine attack status
    grouped_labels = [] 

    # Iterate through the batch with a step of SEQUENCE_LENGTH
    # We drop the remainder if the batch isn't perfectly divisible
    for i in range(0, len(packets) - (len(packets) % SEQUENCE_LENGTH), SEQUENCE_LENGTH):
        # 1. Slice the chunk
        chunk_pkts = packets[i : i + SEQUENCE_LENGTH]
        chunk_t1 = type1[i : i + SEQUENCE_LENGTH]
        chunk_t2 = type2[i : i + SEQUENCE_LENGTH]

        # 2. Join packets with separator
        joined_sequence = SEPARATOR.join(chunk_pkts)
        grouped_packets.append(joined_sequence)

        # 3. Determine Label
        # If ANY packet has type_1 != 0 or type_2 != 0, it's an attack
        is_attack = any(t != 0 for t in chunk_t1) or any(t != 0 for t in chunk_t2)
        grouped_labels.append(1 if is_attack else 0)

    return {"packet": grouped_packets, "labels": grouped_labels}

def tokenize_and_mask(example_batch):
    # 1. Tokenize inputs (The grouped sequences)
    model_inputs = tokenizer(
        example_batch["packet"], 
        truncation=True, 
        padding="max_length", # Pad to ensure uniform tensor shapes
        max_length=MAX_TOKEN_LENGTH
    )
    
    # 2. Tokenize targets (Copy of input - Clean reconstruction)
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            example_batch["packet"], 
            truncation=True, 
            padding="max_length",
            max_length=MAX_TOKEN_LENGTH
        )
    
    # 3. Apply MASKING
    mask_token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    batch_input_ids = model_inputs["input_ids"]
    masked_batch = []
    
    for input_ids in batch_input_ids:
        arr = np.array(input_ids)
        # Create random probability matrix
        probability_matrix = np.random.rand(*arr.shape)
        
        # Avoid masking special tokens (padding=0, eos=1 usually) if necessary
        # For ByT5, 0 is often padding. Let's ensure we don't mask padding if possible, 
        # though the model handles it.
        # Simple mask logic:
        masked_indices = probability_matrix < MASK_PROBABILITY
        
        # Apply mask
        arr[masked_indices] = mask_token_id
        masked_batch.append(arr.tolist())
        
    model_inputs["input_ids"] = masked_batch
    
    # Set -100 for padding tokens in labels so they are ignored by loss function
    labels = target_encodings["input_ids"]
    labels = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels
    ]
    model_inputs["labels"] = labels
    
    return model_inputs

# --- MAIN EXECUTION ---

# Initialize
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
tokenizer = ByT5Tokenizer.from_pretrained(model_checkpoint)

dataset = datasets.load_dataset(
    "csv", 
    sep=",", 
    names=COLUMN_NAMES, 
    data_files={"train": [train_file], "valid": [valid_file]}
)

print(f"Original train size (packets): {len(dataset['train'])}")

# 1. Convert Hex to Bytes (Row by row)
print("Converting Hex to Bytes...")
dataset = dataset.map(hex_to_bytes)

# 2. Group into Sequences (Batched)
# We use a large batch size to minimize data dropped at chunk boundaries
print(f"Grouping into sequences of length {SEQUENCE_LENGTH}...")
dataset = dataset.map(
    group_into_sequences,
    batched=True,
    batch_size=1000, 
    remove_columns=COLUMN_NAMES # Remove old columns, we create new 'packet' and 'labels'
)

print(f"Grouped train size (sequences): {len(dataset['train'])}")

# 3. Tokenize and Mask
print("Tokenizing and Masking...")
dataset = dataset.map(
    tokenize_and_mask, 
    batched=True, 
    num_proc=4, 
    remove_columns=["packet"] # Remove text column, keep tensors
)

# Training Args
finetuned_model_name = f"/home/spritz/storage/disk0/Master_Thesis/Stuff/Byt5/simplified-hex_modbus-sequence_{SEQUENCE_LENGTH}-finetuned"
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
    generation_max_length=MAX_TOKEN_LENGTH,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
    report_to="none",
    push_to_hub=False,
    fp16=False, # Recommended if you have a GPU, speeds up training significantly
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