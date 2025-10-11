import os
import csv
import torch
import random
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments

# ------------------------
# Disable W&B and force TensorBoard logging
# ------------------------
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_LOGGING"] = "tensorboard"

DATA_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset/normal_traffic.txt"

# ------------------------
# Load normal traffic dataset
# ------------------------
inputs, targets = [], []
with open(DATA_PATH, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 1:
            continue
        modbus_frame = row[0].strip()
        if modbus_frame:
            inputs.append(modbus_frame)
            targets.append(modbus_frame)  # target same as input

# ------------------------
# Build Hugging Face dataset and split (90% train, 10% val)
# ------------------------
raw_dataset = Dataset.from_dict({"input": inputs, "target": targets})
train_dataset, val_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42).values()

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

# ------------------------
# Tokenizer & model
# ------------------------
model_name = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

max_length = 384

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
    output_dir="/home/spritz/storage/disk0/Master_Thesis/ByT5/ByT5-project-normalTraffic/byt5_modbus_encoder",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio=0.08,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=8,
    weight_decay=0.08,
    label_smoothing_factor=0.0,  # no smoothing for reconstruction
    num_train_epochs=3,
    fp16=False,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir="/home/spritz/storage/disk0/Master_Thesis/ByT5/ByT5-project-normalTraffic/logs",
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

# Save the final fine-tuned autoencoder
model.save_pretrained("/home/spritz/storage/disk0/Master_Thesis/ByT5/ByT5-project-normalTraffic/byt5_modbus_encoder_final")
tokenizer.save_pretrained("/home/spritz/storage/disk0/Master_Thesis/ByT5/ByT5-project-normalTraffic/byt5_modbus_encoder_final")

# ------------------------
# Evaluate reconstruction
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Show a few reconstruction examples
print("\n🔍 Example reconstructions (input vs output):")
sample_indices = random.sample(range(len(dataset["validation"])), 10)

for i in sample_indices:
    sample_input = dataset["validation"][i]["input"]
    inputs = tokenizer(sample_input, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    reconstructed = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Original:      {sample_input}")
    print(f"Reconstructed: {reconstructed}")
    print("-" * 60)
