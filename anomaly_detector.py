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
    def __init__(self, model_dir, data_path, eval_loss, max_length=MAX_LENGTH):
        self.model_dir = model_dir
        self.data_path = data_path
        self.eval_loss = eval_loss
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
                # take last n layers and average across them and tokens
                hidden_states = outputs.hidden_states[-last_n_layers:]
                hidden_mean = torch.stack(hidden_states).mean(0).mean(1)
                hidden_mean = hidden_mean.cpu().numpy()
                del outputs, hidden_states # free space

            all_embs.append(hidden_mean)
            torch.cuda.empty_cache()   # prevent accumulation
        return np.vstack(all_embs)

    def _compute_reconstruction_error(self, seqs, batch_size=BATCH_SIZE):
        """
        Returns:
            recon_texts: list[str] reconstructed sequences
            errors: np.ndarray reconstruction errors
        """
        self.model.eval()
        recon_texts = []
        errors = []

        for i in tqdm(range(0, len(seqs), batch_size), desc="Reconstructing sequences"):
            batch = seqs[i:i + batch_size]

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

                # 2️⃣ Compute reconstruction error
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                logits = outputs.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = inputs["input_ids"][:, 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                per_token_loss = per_token_loss.view(shift_labels.size(0), -1)
                per_seq_loss = per_token_loss.mean(dim=1).detach().cpu().numpy()

                errors.extend(per_seq_loss)

            torch.cuda.empty_cache()

        return recon_texts, np.array(errors)

    def save_reconstruction_report(self, seqs, labels, filename):
        """
        Save original, reconstructed, label, predicted anomaly, and reconstruction error.
        """
        print(f"\nSaving reconstruction report to {filename} ...")

        recon_texts, recon_errors = self._compute_reconstruction_error(seqs)
        preds = (recon_errors > self.eval_loss).astype(int)  # 1 = anomaly

        with open(filename, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Original", "Reconstructed", "Label", "Predicted", "Error"])

            for orig, recon, y, pred, err in zip(seqs, recon_texts, labels, preds, recon_errors):
                writer.writerow([orig, recon, y, pred, f"{err:.6f}"])

        print(f"Report saved: {filename}")

    def detect(self):
        print("Loading data ...")
        dataset_builder = DatasetBuilder()
        train_pkts, train_lbls, train_tmstmp = dataset_builder._load_packets(data_path=f"{self.data_path}/train.txt")
        train_seqs, train_seqs_lbls = dataset_builder._group_sequences(train_pkts, train_lbls, train_tmstmp)
        test_pckts, test_lbls, test_tmstps = dataset_builder._load_packets(f"{self.data_path}/test.txt")
        test_seqs, test_seqs_lbls = dataset_builder._group_sequences(test_pckts, test_lbls, test_tmstps)
        test_y = np.array(test_seqs_lbls)
        
        if False:
            print("Encoding packets ...")
            train_embeddings = self._get_embeddings_batch(train_seqs)
            #train_y = np.array(train_seqs_lbls)

            test_embeddings = self._get_embeddings_batch(test_seqs)
            test_y = np.array(test_seqs_lbls)

            print("Normalizing embeddings ...")
            scaler = StandardScaler().fit(train_embeddings)
            emb_train_s = scaler.transform(train_embeddings)
            emb_test_s = scaler.transform(test_embeddings)

            print("Dimensional reduction ...")
            pca = PCA(n_components=PCA_DIM, random_state=42).fit(emb_train_s)
            emb_train_reduced = pca.transform(emb_train_s)
            emb_test_reduced = pca.transform(emb_test_s)

            # ---------- 1️⃣ Isolation Forest ----------
            print("\n[IsolationForest] Training on normal sequences only...")
            iso = IsolationForest(
                n_estimators=500,
                max_samples="auto",
                contamination=0.05,   # contamination is the thershold for IF
                max_features=1.0,
                bootstrap=False,
                random_state=42
            ).fit(emb_train_reduced)

            scores_iso = -iso.decision_function(emb_test_reduced)
            preds_iso = np.where(iso.predict(emb_test_reduced) == 1, 0, 1)
            auc_iso = roc_auc_score(test_y, scores_iso)
            f1_iso = f1_score(test_y, preds_iso)
            print(f"IsolationForest → AUC: {auc_iso:.3f}, F1: {f1_iso:.3f}")

            # ---------- 2️⃣ One-Class SVM ----------
            print("\n[OneClassSVM] Training on normal sequences only...")
            ocsvm = OneClassSVM(
                kernel="rbf", 
                gamma="scale", 
                nu=0.03         # nu is the threshold for OCSVM
            ).fit(emb_train_reduced)
            scores_svm = -ocsvm.decision_function(emb_test_reduced)
            preds_svm = np.where(ocsvm.predict(emb_test_reduced) == 1, 0, 1)
            auc_svm = roc_auc_score(test_y, scores_svm)
            f1_svm = f1_score(test_y, preds_svm)
            print(f"OneClassSVM → AUC: {auc_svm:.3f}, F1: {f1_svm:.3f}")

            # ---------- 3️⃣ Reconstruction Error ----------
            print("\n[Reconstruction Error] Evaluating model reconstruction ability...")
            recon_errors = self._compute_reconstruction_error(test_seqs)
            # Compare with eval_loss threshold
            preds_recon = np.where(recon_errors > self.eval_loss, 1, 0)  # 1 = anomaly
            auc_recon = roc_auc_score(test_y, recon_errors)
            f1_recon = f1_score(test_y, preds_recon)
            print(f"ReconstructionError → AUC: {auc_recon:.3f}, F1: {f1_recon:.3f}")

            # ---------- DEBUG ----------
            # Separate normal and attack samples
            normal_indices = np.where(test_y == 0)[0]
            attack_indices = np.where(test_y == 1)[0]

            normal_seqs = [test_seqs[i] for i in normal_indices]
            attack_seqs = [test_seqs[i] for i in attack_indices]

            print("\n[Reconstruction Error] Computing per-class reconstruction errors...")

            # Compute reconstruction errors separately
            normal_errors = self._compute_reconstruction_error(normal_seqs)
            attack_errors = self._compute_reconstruction_error(attack_seqs)

            # Optional: show summary statistics
            print("\nSummary:")
            print(f"Normal   → mean={normal_errors.mean():.4f}, std={normal_errors.std():.4f}")
            print(f"Attack   → mean={attack_errors.mean():.4f}, std={attack_errors.std():.4f}")

                # ---------- SAVE FULL RECONSTRUCTION REPORT ----------
        self.save_reconstruction_report(
            test_seqs,
            test_y,
            filename=f"{self.data_path}/full_reconstruction_report.csv"
        )

        return {
            #"IsolationForest": (auc_iso, f1_iso),
            #"OneClassSVM": (auc_svm, f1_svm),
            #"Reconstruction": (auc_recon, f1_recon)
            #"Perplexity": (auc_perp, f1_perp),
        }