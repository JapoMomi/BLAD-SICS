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
train_file = "/home/spritz/storage/disk0/Master_Thesis/Dataset/splits/train.txt"
valid_file = "/home/spritz/storage/disk0/Master_Thesis/Dataset/splits/validation.txt"

MASK_PROB = 0.15  
# Hyperparameters
SEQUENCE_LENGTH = 1         # Number of packets per sequence
SEPARATOR = " "             # Space separator for byte-level model
MAX_TOKEN_LENGTH = 128     # Fixed length (must match detection)

COLUMN_NAMES = ["packet", "type_1", "type_2", "src", "dst", "timestamp"]

def hex_to_bytes(example):
    """
    Converts Hex String packet to a Latin-1 decoded string.
    This preserves the raw byte values 1-to-1 for the tokenizer.
    """
    hex_str = str(example["packet"])
    try:
        # 1. Convert hex to raw bytes (e.g., "04ab" -> b'\x04\xab')
        byte_data = bytes.fromhex(hex_str)
        # 2. Decode as Latin-1 to get a printable string where char_code == byte_val
        example["packet"] = byte_data.decode("latin-1") 
    except Exception as e:
        # Fallback (should not happen with generated data)
        pass 
    return example

def group_into_sequences(examples):
    """
    Groups packets into sequences of length N.
    """
    packets = examples["packet"]
    type1 = examples["type_1"]
    type2 = examples["type_2"]
    
    grouped_packets = []
    
    # Drop remainder to ensure all sequences are full length
    total_length = len(packets) - (len(packets) % SEQUENCE_LENGTH)
    
    for i in range(0, total_length, SEQUENCE_LENGTH):
        chunk_pkts = packets[i : i + SEQUENCE_LENGTH]
        # Join packets with separator
        joined_sequence = SEPARATOR.join(chunk_pkts)
        grouped_packets.append(joined_sequence)

    return {"packet": grouped_packets}

def tokenize_and_mask(example_batch):
    model_inputs = tokenizer(
        example_batch["packet"], 
        truncation=True, 
        padding="max_length", 
        max_length=MAX_TOKEN_LENGTH
    )
    
    # Clone input_ids for labels
    labels = model_inputs["input_ids"].copy()
    input_ids = model_inputs["input_ids"]
    mask_token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    
    masked_input_ids = []
    
    for i, seq in enumerate(input_ids):
        seq_arr = np.array(seq)
        seq_len = len(seq)
        
        # Determine how many tokens total to mask
        num_tokens_to_mask = int(seq_len * MASK_PROB)
        
        # Create a boolean mask
        mask = np.full(seq_len, False)
        
        # Iteratively add spans until we reach the target mask count
        masked_count = 0
        while masked_count < num_tokens_to_mask:
            # Randomize Span Length (e.g., between 2 and 8 bytes)
            # This is CRITICAL for robustness
            current_span_length = np.random.randint(2, 6) 
            
            # Pick a random start point
            if seq_len - current_span_length <= 0: break
            start = np.random.randint(0, seq_len - current_span_length)
            
            # Apply mask if not already masked
            if not np.any(mask[start : start + current_span_length]):
                mask[start : start + current_span_length] = True
                masked_count += current_span_length
        
        # Don't mask special tokens (0=Pad, 1=EOS)
        special_tokens_mask = [
            1 if token in [tokenizer.pad_token_id, tokenizer.eos_token_id] else 0 
            for token in seq
        ]
        mask = mask & (np.array(special_tokens_mask) == 0)
        
        # Apply mask
        seq_arr[mask] = mask_token_id
        masked_input_ids.append(seq_arr.tolist())

        # Update labels (ignore padding)
        labels[i] = [(l if l != tokenizer.pad_token_id else -100) for l in labels[i]]

    model_inputs["input_ids"] = masked_input_ids
    model_inputs["labels"] = labels
    return model_inputs

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Loading model and tokenizer...")
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    tokenizer = ByT5Tokenizer.from_pretrained(model_checkpoint)

    #dataset = datasets.load_dataset(
    #    "csv", 
    #    sep=",", 
    #    names=COLUMN_NAMES, 
    #    data_files={"train": [train_file], "valid": [valid_file]}
    #)
    print("Loading data via Pandas to avoid SegFault...")
    train_df = pd.read_csv(train_file, names=COLUMN_NAMES, header=None, dtype=str)
    valid_df = pd.read_csv(valid_file, names=COLUMN_NAMES, header=None, dtype=str)
    train_dataset = datasets.Dataset.from_pandas(train_df)
    valid_dataset = datasets.Dataset.from_pandas(valid_df)
    dataset = datasets.DatasetDict({
        "train": train_dataset,
        "valid": valid_dataset
    })
    
    print(f"Loaded Train: {len(dataset['train'])} rows")
    print(f"Loaded Valid: {len(dataset['valid'])} rows")

    # 1. Convert Hex to Latin-1 Bytes
    print("Converting Hex to Raw Bytes (Latin-1)...")
    dataset = dataset.map(hex_to_bytes)
    
    # DEBUG CHECK
    print(f"Sample raw packet (decoded): {repr(dataset['train'][0]['packet'])}")

    # 2. Group into Sequences
    print(f"Grouping into sequences of length {SEQUENCE_LENGTH}...")
    dataset = dataset.map(
        group_into_sequences,
        batched=True,
        batch_size=1000, 
        remove_columns=COLUMN_NAMES 
    )

    # 3. Tokenize and Mask
    print("Tokenizing and Masking...")
    dataset = dataset.map(
        tokenize_and_mask, 
        batched=True, 
        remove_columns=["packet"] 
    )

    # Training Args
    output_path = f"/home/spritz/storage/disk0/Master_Thesis/SingplePacketDetection/Byt5/BYTES_modbus-single_packet-finetuned"
    
    args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        learning_rate=2e-4, 
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        num_train_epochs=15,
        predict_with_generate=True,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        fp16=False, # Set to False if you don't have a GPU
        report_to="none",
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

    print("Starting Training...")
    trainer.train()

    print(f"Saving model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Training Complete.")