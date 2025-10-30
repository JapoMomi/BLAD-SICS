import os
import csv
import torch
import random
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
import sys
# ------------------------
# Disable W&B and force TensorBoard logging
# ------------------------
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_LOGGING"] = "tensorboard"

DATA_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset/normal_traffic.txt"

# ------------------------
# Load normal traffic dataset (packets + timestamps)
# ------------------------
packets, timestamps = [], []
with open(DATA_PATH, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 2:
            continue
        packet = row[0].strip()
        if not packet:
            continue
        try:
            ts = float(row[-1])  # last column is timestamp
        except ValueError:
            continue
        packets.append(packet)
        timestamps.append(ts)

# ------------------------
# Group packets into sequences
# ------------------------
def group_sequences(packets, timestamps, n_packets, max_time_gap):
    sequences = []
    start = 0
    while start < len(packets):
        seq = [packets[start]]
        current_ts = timestamps[start]
        for j in range(start + 1, len(packets)):
            if len(seq) >= n_packets:
                break
            if timestamps[j] - current_ts > max_time_gap:
                break
            seq.append(packets[j])
            current_ts = timestamps[j]
        # Join packets with a separator token
        seq_text = "     ".join(seq)
        sequences.append(seq_text)
        start += 1  # shift by 1 to keep overlapping sequences
    return sequences

# Change this value to test different temporal contexts (3, 4, 5, ...)
N_PACKETS = 4
MAX_TIME_GAP = 2 #seconds
sequences = group_sequences(packets, timestamps, n_packets=N_PACKETS, max_time_gap=MAX_TIME_GAP)

print(f"Generated {len(sequences)} sequences with {N_PACKETS} packets each")

# ------------------------
# Build Dataset for autoencoding
# ------------------------
inputs = sequences
targets = sequences  # same for reconstruction
raw_dataset = Dataset.from_dict({"input": inputs, "target": targets})

# Train/val/test split
train_dataset, temp_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42).values()
val_dataset, test_dataset = temp_dataset.train_test_split(test_size=0.75, seed=42).values()

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# ------------------------
# Tokenizer & model
# ------------------------
model_name = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

max_length = 768  # longer since we have multiple packets

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input"], max_length=max_length, truncation=True
    )
    labels = tokenizer(
        examples["target"], max_length=max_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["input", "target"])

# ------------------------
# Training setup
# ------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir=f"/home/spritz/storage/disk0/Master_Thesis/ByT5/ByT5-project-sequences/byt5_modbus_normalTraf_seq_{N_PACKETS}",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio=0.08,
    per_device_train_batch_size=4,  # smaller due to longer sequences
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    weight_decay=0.08,
    num_train_epochs=3,
    fp16=False,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir="/home/spritz/storage/disk0/Master_Thesis/ByT5/ByT5-project-sequences/logs",
    logging_steps=50,
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ------------------------
# Train
# ------------------------
trainer.train()

# Save fine-tuned sequence autoencoder
save_path = f"/home/spritz/storage/disk0/Master_Thesis/ByT5/ByT5-project-sequences/byt5_modbus_normalTraf_seq_{N_PACKETS}_final"
trainer.save_model(save_path)
#model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# ------------------------
# Evaluate reconstruction
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"\nExample reconstructions for {N_PACKETS}-packet sequences:")
sample_indices = random.sample(range(len(dataset["test"])), 5)

for i in sample_indices:
    seq_input = dataset["test"][i]["input"]
    inputs = tokenizer(seq_input, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=max_length)
    reconstructed = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Original:\n{seq_input}\n")
    print(f"Reconstructed:\n{reconstructed}\n")
    print("-" * 80)
