import random
import datasets
import numpy as np
import pandas as pd
import torch # Added torch for masking logic
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
MASK_PROBABILITY = 0.15 # 15% masking

# Define column names
COLUMN_NAMES = ["packet", "type_1", "type_2", "src", "dst", "timestamp"]
# ---------------------

def filter_normal_packets(data_row):
    """
    Filter to keep only normal packets.
    """
    if data_row["packet"] is None or data_row["packet"] == "":
        return False
    if data_row["type_1"] == 0 and data_row["type_2"] == 0:
        return True
    return False

def convert_to_features(example_batch):
    # 1. Tokenize inputs (Original)
    model_inputs = tokenizer(
        example_batch["packet"], 
        truncation=True, 
        padding=False, 
        max_length=max_len
    )
    
    # 2. Tokenize targets (Original - used as Ground Truth)
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            example_batch["packet"], 
            truncation=True, 
            padding=False, 
            max_length=max_len
        )
    
    # 3. Apply MASKING to model_inputs["input_ids"]
    # We use <extra_id_0> (sentinel) to replace random bytes
    mask_token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    
    batch_input_ids = model_inputs["input_ids"]
    masked_batch = []
    
    for input_ids in batch_input_ids:
        # Convert list to array for easier manipulation
        arr = np.array(input_ids)
        
        # Create a mask of the same shape
        # Avoid masking special tokens if any (though ByT5 is mostly raw bytes)
        # We generate a random matrix
        probability_matrix = np.random.rand(*arr.shape)
        
        # Create boolean mask where prob < 0.15
        masked_indices = probability_matrix < MASK_PROBABILITY
        
        # Apply mask
        arr[masked_indices] = mask_token_id
        masked_batch.append(arr.tolist())
        
    # Update inputs with masked version
    model_inputs["input_ids"] = masked_batch
    
    # Labels are the ORIGINAL (Unmasked) sequences
    model_inputs["labels"] = target_encodings["input_ids"]
    
    return model_inputs

# Initialize Model and Tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
tokenizer = ByT5Tokenizer.from_pretrained(model_checkpoint)

max_length = 128 
max_len = max_length

# Load Dataset
dataset = datasets.load_dataset(
    "csv", 
    sep=",", 
    names=COLUMN_NAMES, 
    data_files={"train": [train_file], "valid": [valid_file]}
)

# 1. Filter: Keep only Normal Packets
print(f"Original train size: {len(dataset['train'])}")
dataset = dataset.filter(filter_normal_packets)
print(f"Filtered (Normal only) train size: {len(dataset['train'])}")

# 2. Preprocess: Tokenize & Apply Masking
dataset = dataset.map(
    convert_to_features, 
    batched=True, 
    num_proc=4, 
    remove_columns=COLUMN_NAMES 
)

# Metrics
batch_size = 16 
model_name = "/home/spritz/storage/disk0/Master_Thesis/Stuff/Byt5/simplified-modbus-reconstruction"
num_epochs = 5 
learning_rate = 1e-4 
gradient_accumulation_steps = 2

finetuned_model_name = f"{model_name}-finetuned"

args = Seq2SeqTrainingArguments(
    output_dir=finetuned_model_name,
    overwrite_output_dir=True,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    weight_decay=0.01,
    log_level="info",
    seed=42,
    dataloader_drop_last=True,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    logging_strategy="steps",
    logging_steps=50,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    generation_max_length=max_length,
    report_to="none", 
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    matches = [1 if p == l else 0 for p, l in zip(decoded_preds, decoded_labels)]
    accuracy = sum(matches) / len(matches)

    result = {"accuracy": accuracy}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["mean_gen_len"] = np.mean(prediction_lens)

    result = {k: round(v, 4) for k, v in result.items()}
    return result

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Start Training
trainer.train()

# Save
model.save_pretrained(finetuned_model_name)
tokenizer.save_pretrained(finetuned_model_name)