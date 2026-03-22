import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt


MODEL_PATH = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/Byt5/BYTES_modbus-sequence_5_ALLMasked-finetuned"
TRAIN_FILE_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits/train.txt" 
TEST_FILE_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits/test.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

def load_data(filepath):
    print(f"Lettura {filepath}...")
    try:
        df = pd.read_csv(filepath, header=None, dtype=str)
        sequences = df[0].values
        # Assumiamo che l'ultima colonna sia la label, ma per il training la ignoreremo
        labels = df.iloc[:, -1].astype(int).values
        return sequences, labels
    except Exception as e:
        print(f"Errore lettura file: {e}")
        return [], []

def hex_to_latin1(hex_str):
    try:
        clean_hex = hex_str.replace(" ", "")
        return bytes.fromhex(clean_hex).decode('latin-1')
    except:
        return ""

def extract_embeddings(model, tokenizer, sequences, batch_size=32, desc="Extracting"):
    """ Estrae gli embedding usando l'Encoder di ByT5 """
    model.eval()
    all_embeddings = []
    
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    
    # Progress bar per monitorare l'estrazione (può essere lunga)
    for i in tqdm(range(0, len(sequences), batch_size), total=total_batches, desc=desc, unit="batch"):
        batch_hex = sequences[i : i + batch_size]
        batch_latin1 = [hex_to_latin1(s) for s in batch_hex]
        
        inputs = tokenizer(
            batch_latin1, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=512
        ).to(DEVICE)
        
        with torch.no_grad():
            encoder_outputs = model.encoder(
                input_ids=inputs.input_ids, 
                attention_mask=inputs.attention_mask
            )
            last_hidden_state = encoder_outputs.last_hidden_state
            
            # Mean Pooling
            mask = inputs.attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            all_embeddings.append(mean_embeddings.cpu().numpy())
            
    if len(all_embeddings) > 0:
        return np.vstack(all_embeddings)
    else:
        return np.array([])

# --- MAIN FLOW ---

if __name__ == "__main__":
    print(f"--- ANOMALY DETECTION (Training solo su Benigni) ---")
    
    # 1. Caricamento Modello ByT5 (Feature Extractor)
    print(f"Caricamento ByT5 Encoder...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)

    # 2. Caricamento Dati
    # TRAIN: Solo Benigni
    seq_train, y_train = load_data(TRAIN_FILE_PATH)
    # TEST: Misto
    seq_test, y_test = load_data(TEST_FILE_PATH)
    print(f"TRAIN Set (Benign Only): {len(seq_train)} samples")
    print(f"TEST Set (Mixed):        {len(seq_test)} samples (Attacchi: {np.sum(y_test)})")

    # 3. Estrazione Features
    # Nota: Estraiamo features per TUTTO il train set benigno per insegnare la normalità
    print("\n--- Fase 1: Estrazione Embeddings ---")
    X_train = extract_embeddings(model, tokenizer, seq_train, batch_size=BATCH_SIZE, desc="Embedding Train")
    X_test = extract_embeddings(model, tokenizer, seq_test, batch_size=BATCH_SIZE, desc="Embedding Test")

    # 4. Addestramento Isolation Forest
    print("\n--- Fase 2: Addestramento Isolation Forest (Unsupervised) ---")
    # contamination='auto' lascia decidere all'algoritmo, oppure puoi settare un valore basso (es. 0.01)
    # se sai che il training set è sporco. Se è pulito, 'auto' va bene.
    clf = IsolationForest(n_estimators=100, contamination='auto', n_jobs=-1, random_state=42)
    
    # NOTA: Qui passiamo SOLO X_train, senza y_train! 
    # L'algoritmo non sa che sono "0", sa solo che "questi sono dati normali".
    clf.fit(X_train)
    print("Modello addestrato sulla 'Normalità'.")

    # 5. Detection su Test Set
    print("\n--- Fase 3: Valutazione su Dati Misti ---")
    # Isolation Forest ritorna: 1 per INLIERS (Benigni), -1 per OUTLIERS (Attacchi)
    raw_preds = clf.predict(X_test)
    
    # Dobbiamo convertire l'output di Isolation Forest (1/-1) nel formato delle tue label (0/1)
    # IF Output:  1 (Normal) -> Tua Label: 0
    # IF Output: -1 (Anomaly) -> Tua Label: 1
    preds_converted = np.where(raw_preds == 1, 0, 1)

    print("\nREPORT DI CLASSIFICAZIONE (Isolation Forest):")
    print(classification_report(y_test, preds_converted, digits=4, target_names=["Benign", "Attack"]))
    
    cm = confusion_matrix(y_test, preds_converted)
    print(f"Confusion Matrix:\n[TP (Attacchi presi): {cm[1][1]} | FN (Persi): {cm[1][0]}]\n[FP (Falsi allarmi): {cm[0][1]} | TN (Benigni OK): {cm[0][0]}]")
    
    # Anomaly Score (più è basso, più è anomalo)
    # Isolation Forest dà score negativi per anomalie e positivi per normali
    # Invertiamo il segno per avere "Score di Anomalia" (Più alto = Più anomalo) per coerenza grafica
    anomaly_scores = -clf.decision_function(X_test)
    
    auc_score = roc_auc_score(y_test, anomaly_scores)
    print(f"ROC AUC: {auc_score:.4f}")

    # Visualizzazione
    plt.figure(figsize=(10, 6))
    plt.hist(anomaly_scores[y_test==0], bins=50, alpha=0.5, label='Benign', color='green', density=True)
    plt.hist(anomaly_scores[y_test==1], bins=50, alpha=0.5, label='Attack', color='red', density=True)
    plt.title('Distribuzione Anomaly Score (Isolation Forest)')
    plt.xlabel('Score Anomalia (Alto = Probabile Attacco)')
    plt.legend()
    plt.savefig("isolation_forest_separation.png")
    print("Grafico salvato in isolation_forest_separation.png")