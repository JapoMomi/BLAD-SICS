import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import ByT5Tokenizer, T5ForConditionalGeneration
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE ---
# Assicurati che questo path punti al modello TimeContext addestrato
MODEL_PATH = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/Byt5/BYTES_modbus-sequence_5_ALLMasked-finetuned"

# File originali
TRAIN_FILE_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits/train.txt" # Solo Benigni
TEST_FILE_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits/test.txt"   # Misto (Benigni + Attacchi)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

def load_data(filepath):
    """Legge il file CSV-like e ritorna liste di sequenze e label"""
    print(f"Lettura {filepath}...")
    df = pd.read_csv(filepath, header=None, dtype=str)
    # Col 0: Sequenza Hex
    sequences = df[0].values
    # Ultima colonna: Label della sequenza (0 o 1)
    labels = df.iloc[:, -1].astype(int).values
    return sequences, labels

def hex_to_latin1(hex_str):
    try:
        clean_hex = hex_str.replace(" ", "")
        return bytes.fromhex(clean_hex).decode('latin-1')
    except:
        return ""

def extract_embeddings(model, tokenizer, sequences, batch_size=32, desc="Extracting"):
    """
    Usa l'Encoder di ByT5 per trasformare le sequenze in vettori numerici (Embeddings).
    Include una progress bar (tqdm).
    """
    model.eval()
    all_embeddings = []
    
    # Calcoliamo il numero totale di batch per la progress bar
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    
    print(f"Inizio estrazione features per {len(sequences)} campioni...")
    
    # tqdm avvolge il range per mostrare la barra
    for i in tqdm(range(0, len(sequences), batch_size), total=total_batches, desc=desc, unit="batch"):
        batch_hex = sequences[i : i + batch_size]
        batch_latin1 = [hex_to_latin1(s) for s in batch_hex]
        
        # Tokenizzazione
        inputs = tokenizer(
            batch_latin1, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=512
        ).to(DEVICE)
        
        with torch.no_grad():
            # USIAMO SOLO L'ENCODER
            encoder_outputs = model.encoder(
                input_ids=inputs.input_ids, 
                attention_mask=inputs.attention_mask
            )
            last_hidden_state = encoder_outputs.last_hidden_state
            
            # MEAN POOLING: Media dei token validi
            mask = inputs.attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            # Portiamo su CPU e accumuliamo
            all_embeddings.append(mean_embeddings.cpu().numpy())
            
    return np.vstack(all_embeddings)

# --- MAIN FLOW ---

if __name__ == "__main__":
    # 1. Caricamento Modello
    print(f"Caricamento ByT5 Encoder da {MODEL_PATH} su {DEVICE}...")
    tokenizer = ByT5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)

    # 2. Caricamento Dati Grezzi
    seq_train_benign, y_train_benign = load_data(TRAIN_FILE_PATH)
    seq_test_mixed, y_test_mixed = load_data(TEST_FILE_PATH)

    # 3. Preparazione Dataset per il Classificatore (Bilanciamento)
    # Prendiamo il 50% del Test Set (che contiene attacchi) per insegnare al classificatore cos'è un attacco
    print("Splitting del Test Set per il training del classificatore...")
    seq_test_split_train, seq_test_split_eval, y_test_split_train, y_test_split_eval = train_test_split(
        seq_test_mixed, y_test_mixed, test_size=0.5, random_state=42, stratify=y_test_mixed
    )

    # Costruiamo il Training Set finale: Benigni originali + Metà del Test Set (misto)
    final_train_seq = np.concatenate([seq_train_benign, seq_test_split_train])
    final_train_y = np.concatenate([y_train_benign, y_test_split_train])

    # Il set di valutazione è la rimanente metà del test set (dati mai visti dal Random Forest)
    final_eval_seq = seq_test_split_eval
    final_eval_y = y_test_split_eval

    print(f"\n--- Setup Dataset Finale ---")
    print(f"TRAIN Set: {len(final_train_seq)} samples (Benigni: {sum(final_train_y==0)}, Attacchi: {sum(final_train_y==1)})")
    print(f"EVAL Set:  {len(final_eval_seq)} samples (Benigni: {sum(final_eval_y==0)}, Attacchi: {sum(final_eval_y==1)})")

    # 4. Estrazione Features (con progress bar!)
    print("\n--- Fase 1: Estrazione Embeddings (ByT5) ---")
    X_train = extract_embeddings(model, tokenizer, final_train_seq, batch_size=BATCH_SIZE, desc="Train Set")
    X_eval = extract_embeddings(model, tokenizer, final_eval_seq, batch_size=BATCH_SIZE, desc="Eval Set")

    print(f"Shape Embeddings Train: {X_train.shape}")

    # 5. Addestramento Classificatore
    print("\n--- Fase 2: Addestramento Random Forest ---")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_train, final_train_y)
    print("Addestramento completato.")

    # 6. Valutazione
    print("\n--- Fase 3: Valutazione ---")
    preds = clf.predict(X_eval)

    print("\nREPORT DI CLASSIFICAZIONE:")
    print(classification_report(final_eval_y, preds, digits=4))
    
    cm = confusion_matrix(final_eval_y, preds)
    print(f"Confusion Matrix:\n[TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}]\n[FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}]")
    # Calcoliamo le probabilità: colonna 0 = Benigno, colonna 1 = Attacco
    probs = clf.predict_proba(X_eval)[:, 1]
    auc_score = roc_auc_score(final_eval_y, probs)
    print(f"AUC: {auc_score:.4f}")

    # (Opzionale) Visualizzazione separazione
    probs = clf.predict_proba(X_eval)[:, 1]
    plt.figure(figsize=(10, 6))
    plt.hist(probs[final_eval_y==0], bins=50, alpha=0.5, label='Benign', color='green', density=True)
    plt.hist(probs[final_eval_y==1], bins=50, alpha=0.5, label='Attack', color='red', density=True)
    plt.title('Probabilità Random Forest (Feature Extraction)')
    plt.xlabel('Probabilità di Attacco')
    plt.legend()
    plt.savefig("classifier_separation.png")
    print("Grafico salvato in classifier_separation.png")