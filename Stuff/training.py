import random
import datasets
import numpy as np
import pandas as pd
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

# Define column names based on your description
# 1: packet, 2: type1, 3: type2, 4: src, 5: dst, 6: timestamp
COLUMN_NAMES = ["packet", "type_1", "type_2", "src", "dst", "timestamp"]
# ---------------------

def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    print(df.head())

def filter_normal_packets(data_row):
    """
    Filter to keep only normal packets.
    Assumes 0 means normal for both type columns.
    """
    # Check for None/Empty
    if data_row["packet"] is None or data_row["packet"] == "":
        return False
        
    # Check if packet is "Normal" (0)
    # Adjust logic if only one column matters
    if data_row["type_1"] == 0 and data_row["type_2"] == 0:
        return True
        
    return False

def convert_to_features(example_batch):
    # INPUT: The packet string
    model_inputs = tokenizer(
        example_batch["packet"], 
        truncation=True, 
        padding=False, 
        max_length=max_len
    )
    
    # TARGET: The same packet string (Reconstruction task)
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            example_batch["packet"], 
            truncation=True, 
            padding=False, 
            max_length=max_len
        )
        
    model_inputs["labels"] = target_encodings["input_ids"]
    return model_inputs

# Initialize Model and Tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
tokenizer = ByT5Tokenizer.from_pretrained(model_checkpoint)

max_length = 128 # Reduced from 512 as packets look shorter (~30 chars)
max_len = max_length

# Load Dataset with column names
dataset = datasets.load_dataset(
    "csv", 
    sep=",", # Assuming CSV implies comma separated based on your example
    names=COLUMN_NAMES, # Manually assign names
    data_files={"train": [train_file], "valid": [valid_file]}
)

# 1. Filter: Keep only Normal Packets
print(f"Original train size: {len(dataset['train'])}")
dataset = dataset.filter(filter_normal_packets)
print(f"Filtered (Normal only) train size: {len(dataset['train'])}")

# 2. Preprocess: Tokenize Input=Packet, Label=Packet
dataset = dataset.map(
    convert_to_features, 
    batched=True, 
    num_proc=4, # Reduced proc for stability
    remove_columns=COLUMN_NAMES # Remove raw columns after tokenization
)

# Metrics
batch_size = 16 # Increased batch size slightly as packets are short
model_name = "simplified-modbus-reconstruction"
num_epochs = 10 
learning_rate = 1e-4 # Increased slightly for reconstruction
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
    report_to="none", # Change to "wandb" or "tensorboard" if needed
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

    # Replace -100 in the predictions as we can't decode them.
    # This is the line that fixes the ValueError
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # Calculate Exact Match Accuracy (More useful for packets than BLEU)
    matches = [1 if p == l else 0 for p, l in zip(decoded_preds, decoded_labels)]
    accuracy = sum(matches) / len(matches)

    result = {"accuracy": accuracy}

    # Length sanity check
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