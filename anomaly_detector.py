import csv
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

from dataset_builder import DatasetBuilder
from config import MAX_LENGTH, BATCH_SIZE, LAST_N_LAYERS, PCA_DIM


class AnomalyDetector:
    def __init__(self, model_dir, data_path, max_length=MAX_LENGTH):
        self.model_dir = model_dir
        self.data_path = data_path
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model from {model_dir} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True).to(self.device)
        self.encoder = self.model.get_encoder()
        self.model.eval()

    def _get_embeddings_batch(self, sequences, batch_size=BATCH_SIZE, last_n_layers=LAST_N_LAYERS):
        all_embs = []
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Extracting embeddings (batch {batch_size})"):
            batch = sequences[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True,
                            padding=True, max_length=self.max_length).to(self.device)

            with torch.no_grad():
                outputs = self.encoder(**inputs, output_hidden_states=True)
                # Take last n layers
                hidden_states = outputs.hidden_states[-last_n_layers:]
                # Stack layers: [n_layers, batch, seq_len, hidden_dim]
                stacked_layers = torch.stack(hidden_states)
                # Average across the *layers* first
                # Shape becomes: [batch, seq_len, hidden_dim]
                layer_averaged = stacked_layers.mean(dim=0)
                hidden_mean = layer_averaged.mean(dim=1)
                hidden_mean = hidden_mean.cpu().numpy()
                del outputs, hidden_states
            
            all_embs.append(hidden_mean)
            torch.cuda.empty_cache() 
            
        return np.vstack(all_embs)

    def _compute_reconstruction_error(self, seqs, batch_size=BATCH_SIZE):
        self.model.eval()
        recon_texts = []
        errors = []

        for i in tqdm(range(0, len(seqs), batch_size), desc="Reconstructing sequences"):
            batch = seqs[i:i + batch_size]

            # Tokenize input batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                # 1️⃣ Generate reconstruction (text output)
                generated = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=1,
                    do_sample=False
                )
                decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                recon_texts.extend(decoded)

                # 2️⃣ Compute reconstruction error (MSE)
                # Encode original inputs
                orig_outputs = self.model.encoder(**inputs)
                orig_embeds = orig_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

                # Encode reconstructed inputs
                recon_inputs = self.tokenizer(
                    decoded,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length
                ).to(self.device)

                recon_outputs = self.model.encoder(**recon_inputs)
                recon_embeds = recon_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

                # Align sequence lengths (pad/truncate to same length)
                min_len = min(orig_embeds.size(1), recon_embeds.size(1))
                orig_embeds = orig_embeds[:, :min_len, :]
                recon_embeds = recon_embeds[:, :min_len, :]

                # Compute per-sequence MSE
                mse = torch.mean((orig_embeds - recon_embeds) ** 2, dim=(1, 2))  # [batch]
                errors.extend(mse.detach().cpu().numpy())

            torch.cuda.empty_cache()

        return recon_texts, np.array(errors)


    def save_reconstruction_report(self, threshold, seqs, labels, recon_texts, recon_errors, filename):
        print(f"\nSaving reconstruction report to {filename} ...")
        preds = (recon_errors > threshold).astype(int)  # 1 = anomaly
        with open(filename, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Original", "Reconstructed", "Label", "Predicted", "Error"])

            for orig, recon, y, pred, err in zip(seqs, recon_texts, labels, preds, recon_errors):
                writer.writerow([orig, recon, y, pred, f"{err:.9f}"])
        print(f"Report saved: {filename}")

    def detect(self):
        print("Loading data ...")
        dataset_builder = DatasetBuilder()
        # Train
        train_pkts, train_lbls, train_tmstmp = dataset_builder._load_packets(data_path=f"{self.data_path}/train.txt")
        train_seqs, train_seqs_lbls = dataset_builder._group_sequences(train_pkts, train_lbls, train_tmstmp)
        # Validation
        val_pkts, val_lbls, val_tmstmp = dataset_builder._load_packets(data_path=f"{self.data_path}/validation.txt")
        val_seqs, val_seqs_lbls = dataset_builder._group_sequences(val_pkts, val_lbls, val_tmstmp)
        # Test
        test_pckts, test_lbls, test_tmstps = dataset_builder._load_packets(f"{self.data_path}/test.txt")
        test_seqs, test_seqs_lbls = dataset_builder._group_sequences(test_pckts, test_lbls, test_tmstps)
        test_y = np.array(test_seqs_lbls)
        
        if False:
            print("Encoding packets ...")
            train_embeddings = self._get_embeddings_batch(train_seqs)
            #train_y = np.array(train_seqs_lbls)
            val_embeddings = self._get_embeddings_batch(val_seqs)
            test_embeddings = self._get_embeddings_batch(test_seqs)
            test_y = np.array(test_seqs_lbls)

            #print("Normalizing embeddings ...")
            #scaler = StandardScaler().fit(train_embeddings)
            #emb_train_s = scaler.transform(train_embeddings)
            #emb_val_s = scaler.transform(val_embeddings)
            #emb_test_s = scaler.transform(test_embeddings)

            #print("Dimensional reduction ...")
            #pca = PCA(n_components=PCA_DIM, random_state=42).fit(train_embeddings)
            #emb_train_reduced = pca.transform(train_embeddings)
            #emb_val_reduced = pca.transform(val_embeddings)
            #emb_test_reduced = pca.transform(test_embeddings)

            # ---------- 1️⃣ Isolation Forest ----------
            print("\n[IsolationForest] Training on normal sequences only...")
            # 1. Fit on Train
            iso = IsolationForest(
                n_estimators=300,
                max_samples=0.8,
                contamination="auto",
                max_features=1,
                bootstrap=True,
                random_state=42
            ).fit(train_embeddings)
            # 2. Get scores for Validation (Normal)
            # Sklearn returns negative scores (higher = normal), so we invert them
            val_scores = -iso.decision_function(val_embeddings)
            # 3. Set Threshold (e.g., 99th percentile of benign validation data)
            # This aligns with the paper's method 
            threshold = np.percentile(val_scores, 99)
            # 4. Detect on Test
            scores_iso = -iso.decision_function(test_embeddings)
            preds_iso = (scores_iso > threshold).astype(int)
            auc_iso = roc_auc_score(test_y, scores_iso)
            f1_iso = f1_score(test_y, preds_iso)
            print(f"IsolationForest → AUC: {auc_iso:.3f}, F1: {f1_iso:.3f}")

            # ---------- 2️⃣ One-Class SVM ----------
            print("\n[OneClassSVM] Training on normal sequences only...")
            ocsvm = OneClassSVM(
                kernel="rbf", 
                gamma="auto", 
                nu=0.01    # Allow only ~1% false positives on train data      
            ).fit(train_embeddings)
             
            scores_svm = -ocsvm.decision_function(test_embeddings)
            preds_svm = np.where(ocsvm.predict(test_embeddings) == 1, 0, 1)
            auc_svm = roc_auc_score(test_y, scores_svm)
            f1_svm = f1_score(test_y, preds_svm)
            print(f"OneClassSVM → AUC: {auc_svm:.3f}, F1: {f1_svm:.3f}")

        if True:
            # ---------- 3️⃣ Reconstruction Error ----------
            print("\n[Reconstruction Error] Evaluating model reconstruction ability...")
            val_recon_text, val_recon_errors = self._compute_reconstruction_error(val_seqs)
            threshold = val_recon_errors.mean() + 3*val_recon_errors.std()
            test_recon_text, test_recon_errors = self._compute_reconstruction_error(test_seqs)
            # Compare with eval_loss threshold
            preds_recon = np.where(test_recon_errors > threshold, 1, 0)  # 1 = anomaly
            auc_recon = roc_auc_score(test_y, test_recon_errors)
            f1_recon = f1_score(test_y, preds_recon)
            print(f"ReconstructionError → AUC: {auc_recon:.3f}, F1: {f1_recon:.3f}")

            # ---------- SAVE FULL RECONSTRUCTION REPORT ----------
            self.save_reconstruction_report(
                threshold,
                test_seqs,
                test_y,
                test_recon_text,
                test_recon_errors,
                filename="full_reconstruction_report.csv"
            )

        return {
            #"IsolationForest": (auc_iso, f1_iso),
            #"OneClassSVM": (auc_svm, f1_svm),
            "Reconstruction": (auc_recon, f1_recon)
        }