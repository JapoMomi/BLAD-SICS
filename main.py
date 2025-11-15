import os

from dataset_builder import DatasetBuilder
from byt5_trainer import ByT5Trainer
from anomaly_detector import AnomalyDetector
from config import DATA_PATH, OUTPUT_DIR, LOGS_DIR, MODEL_NAME, N_PACKETS


if __name__ == "__main__":
    model_dir = os.path.join(OUTPUT_DIR, f"byt5_seq_{N_PACKETS}_normalTraffic_final")
    eval_loss_file = os.path.join(OUTPUT_DIR, f"byt5_seq_{N_PACKETS}_normalTraffic_final_EvalLoss.txt")

    if os.path.exists(model_dir) and os.path.exists(eval_loss_file):
        print(f"\nModel already exists at {model_dir}. Skipping training...")
        # Load eval_loss from file
        with open(eval_loss_file, "r") as f:
            line = f.readline().strip()
            try:
                eval_loss = float(line.split("=")[-1].strip())
                print(f"Loaded eval_loss = {eval_loss}")
            except ValueError:
                print("⚠️ Warning: could not parse eval_loss file, setting to None.")
                eval_loss = None         
    else:    
        # Step 1: Build dataset
        builder = DatasetBuilder()
        tokenized_datasets, raw_dataset, tokenizer = builder.build_dataset(DATA_PATH)

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

        # Step 3: Save eval_loss to file
        with open(eval_loss_file, "w") as f:
            f.write(f"eval_loss = {eval_loss}\n")
        print(f"💾 Eval loss saved to: {eval_loss_file}")

    # Step 3: Anomaly detection phase
    print("\n🚨 Starting anomaly detection phase ...")
    detector = AnomalyDetector(
        model_dir=f"{OUTPUT_DIR}/byt5_seq_{N_PACKETS}_normalTraffic_final",
        data_path=DATA_PATH,
        eval_loss=eval_loss
    )

    results = detector.detect()
    print("\n📊 Summary of anomaly detection results:")
    for method, (auc, f1) in results.items():
        print(f" → {method}: AUC={auc:.3f}, F1={f1:.3f}")
