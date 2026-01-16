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

def tokenize_and_mask_every_packet(example_batch):
    """
    Strategia 'All-Positions': Per ogni sequenza, genera N esempi di training,
    mascherando ogni pacchetto a turno (P1, poi P2, poi P3...).
    Inoltre gestisce la decodifica Hex -> Latin1 in modo sicuro.
    """
    input_strings = []
    label_strings = []
    
    # example_batch["packet"] contiene le stringhe RAW HEX originali
    for hex_seq_str in example_batch["packet"]:
        
        # 1. Split sicuro sulla stringa HEX (lo spazio qui è solo separatore)
        hex_packets = hex_seq_str.strip().split(SEPARATOR)
        
        if len(hex_packets) < 2:
            continue
            
        # 2. Decodifica: Convertiamo i pacchetti da Hex a Latin-1 Bytes
        # (Lo facciamo dopo lo split per evitare il bug del byte 0x20 che diventa spazio)
        packets = []
        try:
            for hp in hex_packets:
                packets.append(bytes.fromhex(hp).decode('latin-1'))
        except ValueError:
            continue # Salta righe corrotte
            
        # 3. Generazione delle Variazioni (Data Augmentation)
        # Invece di un random, facciamo un loop su TUTTI i pacchetti
        for i in range(len(packets)):
            
            # --- INPUT: Maschera il pacchetto i-esimo ---
            masked_packets = packets.copy()
            # Note: See ByT5 warning below regarding <extra_id_0>
            masked_packets[i] = "<extra_id_0>" 
            input_str = SEPARATOR.join(masked_packets)
            
            # --- LABEL: Il contenuto del pacchetto i-esimo ---
            target_packet = packets[i]
            label_str = f"<extra_id_0> {target_packet} <extra_id_1>"
            
            input_strings.append(input_str)
            label_strings.append(label_str)

    # 4. Tokenizzazione massiva
    # Nota: input_strings sarà molto più lunga del batch originale (moltiplicata per seq_len)
    model_inputs = tokenizer(input_strings, max_length=MAX_TOKEN_LENGTH, padding="max_length", truncation=True)
    labels = tokenizer(label_strings, max_length=MAX_TOKEN_LENGTH, padding="max_length", truncation=True)
    
    # Gestione padding nelle labels (-100 per ignorare nella loss)
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Loading model and tokenizer...")
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    tokenizer = ByT5Tokenizer.from_pretrained(model_checkpoint)

    print("Loading Data via Pandas...")
    train_df = pd.read_csv(train_file, names=COLUMN_NAMES, header=None, dtype=str)
    valid_df = pd.read_csv(valid_file, names=COLUMN_NAMES, header=None, dtype=str)
    
    print(f"Original Size -> Train: {len(train_df)}, Valid: {len(valid_df)}")
    
    # Prendiamo il 20% (frac=0.2) delle righe in modo casuale.
    # random_state=42 assicura che se rilanci lo script prendi sempre le stesse righe (riproducibilità).
    train_df = train_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
    valid_df = valid_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
    print(f"Sampled Size (1/5) -> Train: {len(train_df)}, Valid: {len(valid_df)}")

    train_dataset = datasets.Dataset.from_pandas(train_df)
    valid_dataset = datasets.Dataset.from_pandas(valid_df)
    
    dataset = datasets.DatasetDict({"train": train_dataset, "valid": valid_dataset})

    # 2. Tokenize e Mask (Generating All Variations)
    print("Applying All-Packet Masking (Dataset size will increase)...")
    dataset = dataset.map(
        tokenize_and_mask_every_packet, 
        batched=True, 
        remove_columns=COLUMN_NAMES
    )
    
    print(f"Dataset Size After Expansion: Train={len(dataset['train'])}, Valid={len(dataset['valid'])}")

    output_path = f"/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/Byt5/BYTES_modbus-sequence_{SEQUENCE_LENGTH}_ALLMasked-finetuned"
    
    args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        learning_rate=2e-4, 
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        num_train_epochs=15, # Ho abbassato le epoch perché il dataset è 5x più grande!
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