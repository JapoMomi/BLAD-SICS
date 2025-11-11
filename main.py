from dataset_builder import DatasetBuilder
from byt5_trainer import ByT5Trainer

from config import DATA_PATH, OUTPUT_DIR, LOGS_DIR, MODEL_NAME, N_PACKETS


if __name__ == "__main__":
    # Step 1: Build dataset
    builder = DatasetBuilder(data_path=DATA_PATH)
    tokenized_datasets, raw_dataset, tokenizer = builder.build_dataset()

    print(f"Dataset ready: {len(raw_dataset['train'])} train, "
          f"{len(raw_dataset['validation'])} val")

    # Step 2: Train ByT5 model
    trainer = ByT5Trainer(
        model_name=MODEL_NAME,
        tokenizer=tokenizer,
        tokenized_datasets=tokenized_datasets,
        output_dir=OUTPUT_DIR,
        logs_dir=LOGS_DIR,
        n_packets=N_PACKETS
    )

    eval_loss = trainer.train()
    print(f"\n✅ Training complete! Eval loss = {eval_loss}")
