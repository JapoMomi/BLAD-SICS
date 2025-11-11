# byt5_trainer.py
import os
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)

from config import (
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE_TRAIN, BATCH_SIZE_EVAL,
    WEIGHT_DECAY, GRADIENT_ACCUMULATION, FP16, WARMUP_RATIO, LR_SCHEDULER
)


class ByT5Trainer:
    def __init__(self, model_name, tokenizer, tokenized_datasets, output_dir, logs_dir, n_packets):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.tokenized_datasets = tokenized_datasets
        self.output_dir = output_dir
        self.logs_dir = logs_dir
        self.n_packets = n_packets
        self.eval_loss = None

        os.environ["WANDB_DISABLED"] = "true"
        os.environ["HF_LOGGING"] = "tensorboard"

    def train(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=model)

        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}/byt5_seq_{self.n_packets}_normalTraffic",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LEARNING_RATE,
            lr_scheduler_type=LR_SCHEDULER,
            warmup_ratio=WARMUP_RATIO,
            per_device_train_batch_size=BATCH_SIZE_TRAIN,
            per_device_eval_batch_size=BATCH_SIZE_EVAL,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            weight_decay=WEIGHT_DECAY,
            num_train_epochs=NUM_EPOCHS,
            fp16=FP16,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_dir=self.logs_dir,
            logging_steps=50,
            report_to="tensorboard",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # Run training
        trainer.train()

        # Save best model
        save_path = f"{self.output_dir}/byt5_seq_{self.n_packets}_normalTraffic_final"
        trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Store best eval loss
        self.eval_loss = trainer.state.best_metric
        return self.eval_loss
