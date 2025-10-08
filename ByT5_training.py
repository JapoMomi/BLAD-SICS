import os
import csv
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments

# ------------------------
# Disable W&B (force TensorBoard logging)
# ------------------------
os.environ["WANDB_DISABLED"] = "true"   # turn off W&B
os.environ["HF_LOGGING"] = "tensorboard"  # ensure Trainer uses tensorboard

DATA_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset/IanRawDataset.txt"

# ------------------------
# Load raw dataset
# ------------------------
inputs, targets = [], []
with open(DATA_PATH, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 3:
            continue
        modbus_frame = row[0].strip()
        categorization = row[1].strip()
        attack_type = row[2].strip()
        #print(modbus_frame, " ", attack_type)

        inputs.append(modbus_frame)
        targets.append(f"{categorization}")
        #targets.append(f"{categorization},{attack_type}")
        #print(targets)

# Build Hugging Face dataset
raw_dataset = Dataset.from_dict({"input": inputs, "target": targets})
dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)

# ------------------------
# Tokenizer & model
# ------------------------
model_name = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

max_input_length = 256
max_target_length = 16

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input"], max_length=max_input_length, truncation=True
    )
    labels = tokenizer(
        examples["target"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["input", "target"])

# ------------------------
# Training setup
# ------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="/home/spritz/storage/disk0/Master_Thesis/ByT5/ByT5-project/byt5_modbus_small",
    eval_strategy="epoch", # da guardare
    learning_rate=3e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    fp16=False,
    logging_dir="/home/spritz/storage/disk0/Master_Thesis/ByT5/ByT5-project/logs",
    report_to="tensorboard",
    logging_steps=50,                # log every 50 steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ------------------------
# Train
# ------------------------
trainer.train(resume_from_checkpoint=True)
# Save final model
trainer.save_model("/home/spritz/storage/disk0/Master_Thesis/ByT5/ByT5-project/byt5_modbus_small_final")
tokenizer.save_pretrained("/home/spritz/storage/disk0/Master_Thesis/ByT5/ByT5-project/byt5_modbus_small_final")