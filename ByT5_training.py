import os
import csv
import torch
import random
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

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

# ------------------------
# Building Datasets
# ------------------------
raw_dataset = Dataset.from_dict({"input": inputs, "target": targets})
# First split: Train + Temp (Val + Test)
train_dataset, temp_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42).values()
# Second split: Validation + Test (from Temp)
val_dataset, test_dataset = temp_dataset.train_test_split(test_size=0.5, seed=42).values()
# ------------------------
# dataset["train"] → 80%
# dataset["validation"] → 10%
# dataset["test"] → 10%
# ------------------------
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
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,                    
    lr_scheduler_type="cosine",             
    warmup_ratio=0.1,                       
    per_device_train_batch_size=8,          
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,         
    weight_decay=0.05,                      
    num_train_epochs=5,                     
    fp16=False,                             
    save_total_limit=2,
    load_best_model_at_end=True,           
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir="/home/spritz/storage/disk0/Master_Thesis/ByT5/ByT5-project/logs",
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
# Save final model
trainer.save_model("/home/spritz/storage/disk0/Master_Thesis/ByT5/ByT5-project/byt5_modbus_small_final")
tokenizer.save_pretrained("/home/spritz/storage/disk0/Master_Thesis/ByT5/ByT5-project/byt5_modbus_small_final")

# ------------------------
# Test
# ------------------------
print("\nEvaluating on test split...")

# Evaluate loss on test set
test_results = trainer.evaluate(tokenized_datasets["test"])
print("Test set loss and metrics:", test_results)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

predictions, references = [], []

for example in dataset["test"]:
    inputs = tokenizer(example["input"], return_tensors="pt").to(device)  # move inputs to GPU
    outputs = model.generate(**inputs)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    ref = example["target"].strip()
    predictions.append(pred)
    references.append(ref)

# Convert predictions to integers and handle malformed outputs
predictions_clean = []
for p in predictions:
    try:
        val = int(p)
        if val < 0 or val > 7:  # ensure it’s within 0-7
            val = 0  # fallback to normal
    except ValueError:
        val = 0  # fallback to normal
    predictions_clean.append(val)

references_clean = [int(r) for r in references]

# Compute multi-class metrics
precision, recall, f1, _ = precision_recall_fscore_support(
    references_clean, predictions_clean, average="weighted"
)
acc = accuracy_score(references_clean, predictions_clean)

print("\nFinal Test Metrics (Multi-class):")
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

# Confusion matrix
cm = confusion_matrix(references_clean, predictions_clean, labels=list(range(8)))
disp = ConfusionMatrixDisplay(cm, display_labels=[f"Class {i}" for i in range(8)])
disp.plot(cmap="Blues", xticks_rotation="vertical")
plt.title("ByT5-small Modbus RTU Multi-class Intrusion Detection Results")
plt.show()

# Print example predictions
print("\nExample predictions:")
sample_indices = random.sample(range(len(dataset["test"])), 10)
for i in sample_indices:
    print(f"Input: {dataset['test'][i]['input']}")
    print(f"Expected: {references_clean[i]} | Predicted: {predictions_clean[i]}")
    print("-" * 50)