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
train_file = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits/train.txt"
valid_file = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits/validation.txt"

SEQUENCE_LENGTH = 5         
SEPARATOR = " "             
MAX_TOKEN_LENGTH = 512     

COLUMN_NAMES = ["packet", "label"]

def tokenize_and_mask_whole_packet(example_batch):
    """
    Approccio 'High Level': Manipoliamo le stringhe e lasciamo fare al Tokenizer.
    """
    input_strings = []
    label_strings = []
    
    for seq_str in example_batch["packet"]:
        # 1. Split della stringa (i pacchetti sono già convertiti in latin-1 dalla map precedente)
        packets = seq_str.split(SEPARATOR)
        
        if len(packets) < 2: 
            # Fallback per righe vuote o corrotte
            input_strings.append(seq_str)
            label_strings.append("")
            continue
            
        # 2. Scegliamo quale pacchetto mascherare
        idx_to_mask = random.randint(0, len(packets) - 1)
        
        # 3. Costruiamo l'INPUT (Stringa con <extra_id_0>)
        masked_packets = packets.copy()
        masked_packets[idx_to_mask] = "<extra_id_0>" 
        input_str = SEPARATOR.join(masked_packets)
        
        # 4. Costruiamo la LABEL (Stringa target)
        target_packet = packets[idx_to_mask]
        label_str = f"<extra_id_0> {target_packet} <extra_id_1>"
        
        input_strings.append(input_str)
        label_strings.append(label_str)

    # 5. Il Tokenizer converte "<extra_id_0>" -> 258 automaticamente
    model_inputs = tokenizer(input_strings, max_length=MAX_TOKEN_LENGTH, padding="max_length", truncation=True)
    labels = tokenizer(label_strings, max_length=MAX_TOKEN_LENGTH, padding="max_length", truncation=True)
    
    # Gestione padding nelle labels (-100)
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def hex_to_bytes(example):
    """Convert Hex String -> Latin-1 String (Raw Bytes)"""
    hex_str = str(example["packet"])
    try:
        parts = hex_str.split(SEPARATOR)
        decoded_parts = []
        for p in parts:
             decoded_parts.append(bytes.fromhex(p).decode('latin-1'))
        example["packet"] = SEPARATOR.join(decoded_parts)
    except Exception:
        pass 
    return example

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Loading model and tokenizer...")
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    tokenizer = ByT5Tokenizer.from_pretrained(model_checkpoint)

    print("Loading Data via Pandas...")
    train_df = pd.read_csv(train_file, names=COLUMN_NAMES, header=None, dtype=str)
    valid_df = pd.read_csv(valid_file, names=COLUMN_NAMES, header=None, dtype=str)
    
    train_dataset = datasets.Dataset.from_pandas(train_df)
    valid_dataset = datasets.Dataset.from_pandas(valid_df)
    
    dataset = datasets.DatasetDict({"train": train_dataset, "valid": valid_dataset})

    # 1. Convertiamo Hex -> Latin1 String
    print("Converting Hex to Raw Bytes Strings...")
    dataset = dataset.map(hex_to_bytes)

    # 2. Tokenize e Mask (usando il tokenizer)
    print("Applying Whole Packet Masking...")
    dataset = dataset.map(
        tokenize_and_mask_whole_packet, 
        batched=True, 
        remove_columns=COLUMN_NAMES 
    )

    output_path = f"/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/Byt5/BYTES_modbus-sequence_{SEQUENCE_LENGTH}-finetuned"
    
    args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        learning_rate=2e-4, 
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        num_train_epochs=15,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        bf16=True,
        fp16=False, 
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