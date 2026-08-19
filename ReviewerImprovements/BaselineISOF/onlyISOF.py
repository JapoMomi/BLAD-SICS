import csv
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# --- CONFIGURAZIONE PERCORSI ---
VAL_FILE = os.path.join(SCRIPT_DIR, "../../Dataset/singlePacketSplits/validation.txt")
TEST_FILE = os.path.join(SCRIPT_DIR, "../../Dataset/singlePacketSplits/test.txt")

TARGET_FPR = 20.0 

def load_single_packets(filepath):
    """
    Legge i file CSV dei singoli pacchetti.
    Prende il payload esadecimale, rimuove eventuali spazi preesistenti e 
    lo formatta in coppie di byte separate da spazio per il Vectorizer.
    Converte tutte le label > 0 in 1 (Binary Classification).
    """
    print(f"Lettura pacchetti da {filepath}...")
    packets = []
    labels = []
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Salta righe vuote o malformate
            if len(row) < 2:
                continue
            
            payload = row[0].strip()
            try:
                raw_label = int(row[1])
                # Binarizzazione: se è > 0 è un attacco (1), altrimenti è sano (0)
                label = 1 if raw_label > 0 else 0
            except ValueError:
                continue
            
            # Formattazione: "040312" -> "04 03 12"
            payload_clean = payload.replace(" ", "")
            spaced_hex = " ".join([payload_clean[i:i+2] for i in range(0, len(payload_clean), 2)])
            
            packets.append(spaced_hex)
            labels.append(label)
                
    return packets, np.array(labels)

def main():
    print("="*75)
    print(" TRIVIAL BASELINE: BYTE N-GRAM + ISOLATION FOREST (SINGLE PACKET)")
    print("="*75)

    # 1. Caricamento Dati
    val_packets, val_labels = load_single_packets(VAL_FILE)
    test_packets, test_labels = load_single_packets(TEST_FILE)

    # 2. Estrazione delle Feature (Byte 2-grams)
    print("\nEstrazione dei 2-grammi di byte (Feature Engineering Statistica)...")
    # token_pattern r'(?u)\b\w+\b' assicura che legga i singoli byte hex come parole (es. "0a", "ff")
    #vectorizer = CountVectorizer(ngram_range=(2, 2), token_pattern=r'(?u)\b\w+\b')
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), token_pattern=r'(?u)\b\w+\b')

    # Fit sul validation e trasformazione
    X_val = vectorizer.fit_transform(val_packets)
    X_test = vectorizer.transform(test_packets)
    
    print(f"Vocabolario estratto: {len(vectorizer.vocabulary_)} n-grammi unici.")

    # 3. Isolamento del traffico sano per il training Unsupervised
    X_val_sano = X_val[val_labels == 0]

    # 4. Training Isolation Forest
    print("\nAddestramento Isolation Forest (100 stimatori)...")
    clf = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_val_sano)

    # 5. Calcolo Score di Anomalia
    print("Calcolo degli anomaly scores...")
    val_anomaly_scores = -clf.decision_function(X_val_sano)
    test_anomaly_scores = -clf.decision_function(X_test)

    # 6. Ricerca della Soglia all'1% FPR
    print(f"\nCalcolo della soglia al {TARGET_FPR}% FPR sul Validation Set...")
    threshold = np.percentile(val_anomaly_scores, 100.0 - TARGET_FPR)
    print(f"Soglia calcolata: {threshold:.4f}")

    # 7. Valutazione sul Test Set
    preds = (test_anomaly_scores > threshold).astype(int)
    
    # 8. Stampa dei Risultati
    print(f"\n{'='*75}")
    print(f" RISULTATI BASELINE (TEST SET SINGOLI PACCHETTI)")
    print(f"{'='*75}")
    # zero_division=0 gestisce l'eventualità che il modello non preveeda mai la classe Attack
    print(classification_report(test_labels, preds, digits=4, target_names=["Benign", "Attack"], zero_division=0))
    cm = confusion_matrix(test_labels, preds)
    print(f"Confusion Matrix:\n[TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}]\n[FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}]")
    
    try:
        auc = roc_auc_score(test_labels, test_anomaly_scores)
        print(f"ROC AUC: {auc:.4f}")
    except ValueError:
        pass

if __name__ == "__main__":
    main()