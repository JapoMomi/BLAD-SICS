import os

from dataset_builder import DatasetBuilder
from byt5_trainer import ByT5Trainer
from anomaly_detector import AnomalyDetector
from config import DATA_PATH, OUTPUT_DIR, LOGS_DIR, MODEL_NAME, N_PACKETS


if __name__ == "__main__":
    model_dir = os.path.join(OUTPUT_DIR, f"byt5_seq_{N_PACKETS}_normalTraffic_final")
    #eval_loss_file = os.path.join(OUTPUT_DIR, f"byt5_seq_{N_PACKETS}_normalTraffic_final_EvalLoss.txt")

    # Step 1: Build dataset
    builder = DatasetBuilder()
    tokenized_datasets, raw_dataset, tokenizer = builder.build_dataset(DATA_PATH)

    print(f"Dataset ready: {len(raw_dataset['train'])} train, "
        f"{len(raw_dataset['validation'])} val,"
        f"{len(raw_dataset['test'])} test,")

    if os.path.exists(model_dir):
        print(f"\nModel already exists at {model_dir}. Skipping training...")         
    else:    
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

    # Step 3: Anomaly detection phase
    print("\n🚨 Starting anomaly detection phase ...")
    detector = AnomalyDetector(
        model_dir=f"{OUTPUT_DIR}/byt5_seq_{N_PACKETS}_normalTraffic_final",
        data_path=DATA_PATH,
    )

    results = detector.detect()
    print("\n📊 Summary of anomaly detection results:")
    for method, (auc, f1) in results.items():
        print(f" → {method}: AUC={auc:.3f}, F1={f1:.3f}")
