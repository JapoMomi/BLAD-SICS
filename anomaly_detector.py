import csv
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

from dataset_builder import DatasetBuilder
from config import MAX_LENGTH, BATCH_SIZE, LAST_N_LAYERS, PCA_DIM


class AnomalyDetector:
    def __init__(self, model_dir, tokenizer, data_path, eval_loss, max_length=MAX_LENGTH):
        self.model_dir = model_dir
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.eval_loss = eval_loss
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model from {model_dir} ...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device)
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

            all_embs.append(hidden_mean)
            torch.cuda.empty_cache()   # prevent accumulation
        return np.vstack(all_embs)

    def detect(self):
        print("Loading data ...")
        train_pkts, train_lbls, train_tmstmp = DatasetBuilder._load_packets(f"{self.data_path}/train.txt")
        train_seqs, train_seqs_lbls = DatasetBuilder._group_sequences(train_pkts, train_lbls, train_tmstmp)
        test_pckts, test_lbls, test_tmstps = DatasetBuilder._load_packets(f"{self.data_path}/test.txt")
        test_seqs, test_seqs_lbls = DatasetBuilder._group_sequences(test_pckts, test_lbls, test_tmstps)

        print("Encoding packets ...")
        train_embeddings = self._get_embeddings_batch(train_seqs)
        train_y = np.array(train_seqs_lbls)

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