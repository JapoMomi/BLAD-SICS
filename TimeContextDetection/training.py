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

# Parametri
SEQUENCE_LENGTH = 5  
MAX_TOKEN_LENGTH = 512

def tokenize_and_mask_every_packet(example_batch):
    """
    Prende le sequenze di pacchetti (già create nel dataset) e genera 
    esempi di training mascherando un pacchetto alla volta.
    Gestisce:
    - Decoding Hex -> Latin1
    - Mascheramento posizionale (Data Augmentation)
    - Prevenzione errori su batch vuoti
    """
    input_strings = []
    label_strings = []
    
    # example_batch["packet"] contiene la stringa completa della sequenza (es. "P1 P2 P3 P4 P5")
    for hex_seq_str in example_batch["packet"]:
        
        # 1. Controllo validità stringa
        if not hex_seq_str: 
            continue
            
        # 2. Split sicuro: .split() gestisce automaticamente spazi singoli, doppi o tabulazioni
        hex_packets = hex_seq_str.strip().split()
        
        # Se la riga è corrotta e ha meno di 2 pacchetti, la saltiamo
        if len(hex_packets) < 2:
            continue
            
        # 3. Decodifica: Convertiamo i pacchetti da Hex a Latin-1 Bytes
        packets = []
        try:
            for hp in hex_packets:
                packets.append(bytes.fromhex(hp).decode('latin-1'))
        except ValueError:
            continue # Salta intera riga se c'è un pacchetto hex corrotto
            
        # 4. Generazione delle Variazioni
        # Il dataset contiene già la finestra. Noi generiamo 5 esempi per ogni riga,
        # insegnando al modello a predire il pacchetto i-esimo dato il contesto degli altri.
        for i in range(len(packets)):
            
            # --- INPUT: Maschera il pacchetto i-esimo ---
            masked_packets = packets.copy()
            # Usiamo il sentinel token standard. ByT5 lo mapperà internamente al byte corretto (es. 258)
            masked_packets[i] = "<extra_id_0>" 
            
            # Ricostruiamo la stringa con lo spazio come separatore
            input_str = " ".join(masked_packets)
            
            # --- LABEL: Il contenuto del pacchetto i-esimo ---
            target_packet = packets[i]
            label_str = f"<extra_id_0> {target_packet} <extra_id_1>"
            
            input_strings.append(input_str)
            label_strings.append(label_str)

    # 5. Tokenizzazione
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
    # MODIFICA CARICAMENTO:
    # sep="," -> Il file ha virgole che separano la sequenza dalle label numeriche
    # usecols=[0] -> Carichiamo SOLO la prima colonna (la sequenza di pacchetti)
    # names=["packet"] -> Assegniamo il nome per riferirci dopo
    train_df = pd.read_csv(train_file, sep=",", header=None, usecols=[0], names=["packet"], dtype=str)
    valid_df = pd.read_csv(valid_file, sep=",", header=None, usecols=[0], names=["packet"], dtype=str)
    
    # Rimuoviamo eventuali righe vuote o NaN generate dal parsing
    train_df = train_df.dropna()
    valid_df = valid_df.dropna()
    
    print(f"Original Size -> Train: {len(train_df)}, Valid: {len(valid_df)}")
    
    # Sampling (Opzionale: mantieni se vuoi ridurre i tempi di test)
    train_df = train_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
    valid_df = valid_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
    print(f"Sampled Size (1/5) -> Train: {len(train_df)}, Valid: {len(valid_df)}")

    train_dataset = datasets.Dataset.from_pandas(train_df)
    valid_dataset = datasets.Dataset.from_pandas(valid_df)
    
    dataset = datasets.DatasetDict({"train": train_dataset, "valid": valid_dataset})

    # Tokenize e Mask
    print("Applying All-Packet Masking (Dataset size will increase)...")
    dataset = dataset.map(
        tokenize_and_mask_every_packet, 
        batched=True, 
        remove_columns=["packet"] # Rimuoviamo la colonna raw, lasciamo i tensori
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