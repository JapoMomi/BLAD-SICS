import os
import csv
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm
import torch.nn.functional as F

# ======================================================
# CONFIG
# ======================================================
MODEL_PATH = "/home/spritz/storage/disk0/Master_Thesis/ByT5/ByT5-project-sequences/byt5_modbus_normalTraf_seq_3_final"
DATA_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset/IanRawDataset.txt"

N_PACKETS = 3          # sequence length
MAX_GAP = 2            # seconds
MAX_LEN = 768          # max tokens for tokenizer
BATCH_SIZE = 32        # for embedding extraction
PCA_DIM = 128          # compressed embedding size

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

# ======================================================
# LOAD DATASET
# ======================================================
def load_dataset(path):
    pkts, labels, ts = [], [], []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            pkt = row[0].strip()
            try:
                cat = int(row[1].strip())   # 0 = normal, 1 = attack
                timestamp = float(row[-1])
            except ValueError:
                continue
            pkts.append(pkt)
            labels.append(cat)
            ts.append(timestamp)
    return pkts, labels, ts

# ======================================================
# GROUP PACKETS INTO SEQUENCES
# ======================================================
def group_sequences(packets, labels, timestamps, n_packets=4, max_time_gap=0.5):
    sequences, seq_labels = [], []
    start = 0
    while start < len(packets):
        seq = [packets[start]]
        seq_lbls = [labels[start]]
        current_ts = timestamps[start]
        for j in range(start + 1, len(packets)):
            if len(seq) >= n_packets:
                break
            if timestamps[j] - current_ts > max_time_gap:
                break
            seq.append(packets[j])
            seq_lbls.append(labels[j])
            current_ts = timestamps[j]
        seq_text = " <SEP> ".join(seq)
        seq_label = 1 if any(l != 0 for l in seq_lbls) else 0
        sequences.append(seq_text)
        seq_labels.append(seq_label)
        start += 1
    return sequences, seq_labels

# ======================================================
# LOAD MODEL + TOKENIZER
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
encoder = model.get_encoder()

# ======================================================
# BATCHED EMBEDDING EXTRACTION
# ======================================================
def get_embeddings_batch(sequences, batch_size=32):
    all_embs = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings (batched)"):
        batch = sequences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           padding=True, max_length=MAX_LEN).to(device)
        with torch.no_grad():
            out = encoder(**inputs)
            hidden = out.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embs.append(hidden)
    return np.vstack(all_embs)

# ======================================================
# LOAD + GROUP DATA
# ======================================================
pkts, lbls, ts = load_dataset(DATA_PATH)
print(f"Loaded {len(pkts)} packets.")

seqs, seq_labels = group_sequences(pkts, lbls, ts, N_PACKETS, MAX_GAP)
print(f"Built {len(seqs)} sequences ({sum(seq_labels)} attacks, {len(seqs)-sum(seq_labels)} normals)")

# ======================================================
# EMBEDDING EXTRACTION
# ======================================================
embeddings = get_embeddings_batch(seqs, BATCH_SIZE)
y = np.array(seq_labels)

# ======================================================
# PCA COMPRESSION
# ======================================================
print(f"\nApplying PCA → {PCA_DIM} dimensions...")
pca = PCA(n_components=PCA_DIM, random_state=42)
embeddings_reduced = pca.fit_transform(embeddings)

# Split train/test (train = normal only)
emb_train = embeddings_reduced[y == 0]
emb_test = embeddings_reduced
y_test = y

# ======================================================
# 1️⃣ Isolation Forest
# ======================================================
print("\n[IsolationForest] training on normal sequences only...")
iso = IsolationForest(
    n_estimators=300,
    max_samples=0.8,
    contamination=0.1,
    max_features=0.8,
    bootstrap=True,
    random_state=42
).fit(emb_train)

scores_iso = -iso.decision_function(emb_test)
preds_iso = np.where(iso.predict(emb_test) == 1, 0, 1)

auc_iso = roc_auc_score(y_test, scores_iso)
f1_iso = f1_score(y_test, preds_iso)
print(f"IsolationForest → AUC: {auc_iso:.3f}, F1: {f1_iso:.3f}")

# ======================================================
# 2️⃣ One-Class SVM
# ======================================================
print("\n[OneClassSVM] training on normal sequences only...")
svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
svm.fit(emb_train)
scores_svm = -svm.decision_function(emb_test)
preds_svm = np.where(svm.predict(emb_test) == 1, 0, 1)

auc_svm = roc_auc_score(y_test, scores_svm)
f1_svm = f1_score(y_test, preds_svm)
print(f"OneClassSVM → AUC: {auc_svm:.3f}, F1: {f1_svm:.3f}")

# ======================================================
# 3️⃣ Cosine Similarity Anomaly Score
# ======================================================
print("\n[Cosine Similarity] evaluating centroid-based distance...")
centroid = emb_train.mean(axis=0)
scores_cosine = 1 - F.cosine_similarity(
    torch.tensor(emb_test), torch.tensor(centroid).unsqueeze(0), dim=1
).numpy()

auc_cosine = roc_auc_score(y_test, scores_cosine)
f1_cosine = f1_score(y_test, scores_cosine > np.median(scores_cosine))
print(f"CosineSimilarity → AUC: {auc_cosine:.3f}, F1: {f1_cosine:.3f}")

# ======================================================
# 4️⃣ Reconstruction Error (sampled)
# ======================================================
def reconstruction_error(seq):
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=MAX_LEN)
    reconstructed = tokenizer.decode(outputs[0], skip_special_tokens=True)
    L = max(len(seq), len(reconstructed))
    diff = sum(a != b for a, b in zip(seq, reconstructed)) / L
    return diff

print("\nComputing reconstruction error (sampled 200 each for speed)...")
normal_idxs = np.where(y == 0)[0][:200]
attack_idxs = np.where(y == 1)[0][:200]
errors_normal = [reconstruction_error(seqs[i]) for i in tqdm(normal_idxs)]
errors_attack = [reconstruction_error(seqs[i]) for i in tqdm(attack_idxs)]
scores_recon = np.concatenate([errors_normal, errors_attack])
y_recon = np.array([0]*len(errors_normal) + [1]*len(errors_attack))
auc_recon = roc_auc_score(y_recon, scores_recon)
print(f"Reconstruction Error → AUC: {auc_recon:.3f}")

# ======================================================
# 5️⃣ Perplexity-like Score (sampled)
# ======================================================
def pseudo_perplexity(seq):
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    labels = inputs["input_ids"].clone()
    with torch.no_grad():
        loss = model(**inputs, labels=labels).loss.item()
    return loss

print("\nComputing pseudo-perplexity (sampled 200 each for speed)...")
losses_normal = [pseudo_perplexity(seqs[i]) for i in tqdm(normal_idxs)]
losses_attack = [pseudo_perplexity(seqs[i]) for i in tqdm(attack_idxs)]
scores_ppl = np.concatenate([losses_normal, losses_attack])
y_ppl = np.array([0]*len(losses_normal) + [1]*len(losses_attack))
auc_ppl = roc_auc_score(y_ppl, scores_ppl)
print(f"Perplexity-like Score → AUC: {auc_ppl:.3f}")

# ======================================================
# SUMMARY
# ======================================================
print("\n================== SUMMARY ==================")
print(f"IsolationForest   → AUC: {auc_iso:.3f}, F1: {f1_iso:.3f}")
print(f"OneClassSVM       → AUC: {auc_svm:.3f}, F1: {f1_svm:.3f}")
print(f"CosineSimilarity  → AUC: {auc_cosine:.3f}, F1: {f1_cosine:.3f}")
print(f"ReconstructionErr → AUC: {auc_recon:.3f}")
print(f"Perplexity Score  → AUC: {auc_ppl:.3f}")
print("=============================================")
