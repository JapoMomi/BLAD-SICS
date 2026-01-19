import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# --- CONFIGURATION ---
SEQUENCE_LENGTH = 5
MAX_LENGTH = 512
SEPARATOR = ' '

MODEL_PATH = f"/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/Byt5/BYTES_modbus-sequence_{SEQUENCE_LENGTH}_ALLMasked-finetuned"
VAL_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits/validation.txt"
TEST_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits/test.txt"

def hex_to_latin1(hex_sequence):
    """Helper: Hex String -> Latin-1 String"""
    try:
        parts = hex_sequence.split(SEPARATOR)
        decoded_parts = [bytes.fromhex(p).decode('latin-1') for p in parts]
        return SEPARATOR.join(decoded_parts)
    except:
        return hex_sequence

def evaluate_sequence_leave_one_out(model, tokenizer, sequence_str, device):
    """
    Leave-One-Out Evaluation using Log Probabilities (NOT Loss).
    """
    # Splittiamo la stringa HEX originale (qui lo spazio è solo il separatore)
    hex_packets = sequence_str.strip().split(SEPARATOR)
    # Decodifichiamo ogni pacchetto singolarmente
    packets = []
    for h_pack in hex_packets:
        try:
            # Converte da hex a bytes e poi a stringa latin-1
            dec_p = bytes.fromhex(h_pack).decode('latin-1')
            packets.append(dec_p)
        except ValueError:
            # Fallback se c'è sporcizia (es. virgole o newline residui)
            packets.append(h_pack)
    
    # --- CHECK 1: Lunghezza Errata (Causa più probabile dei -100) ---
    if len(packets) != SEQUENCE_LENGTH:
        return -100.0, [-100.0] * SEQUENCE_LENGTH

    individual_scores = []
    input_texts = []
    target_texts = []
    
    # 1. Prepare Batch (5 variations)
    for i in range(SEQUENCE_LENGTH):
        # Input: Mask packet i
        masked_packets = packets.copy()
        masked_packets[i] = "<extra_id_0>"
        input_text = SEPARATOR.join(masked_packets)
        
        # Target: <extra_id_0> packet_content <extra_id_1>
        target_text = f"<extra_id_0> {packets[i]} <extra_id_1>"
        
        input_texts.append(input_text)
        target_texts.append(target_text)
        
    # 2. Tokenize
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
    targets = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)

    # 3. Forward Pass & Probability Calculation
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=targets.input_ids)
        logits = outputs.logits # Shape: [Batch, SeqLen, Vocab]
        
        # Apply Log Softmax to get Log Probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Get the LogProb of the TRUE target token
        # targets.input_ids shape: [Batch, SeqLen] -> unsqueeze to [Batch, SeqLen, 1]
        target_ids = targets.input_ids.unsqueeze(-1)
        
        # Gather: Select the log_prob corresponding to the target_id at each position
        # Shape: [Batch, SeqLen, 1] -> squeeze back to [Batch, SeqLen]
        token_log_probs = log_probs.gather(-1, target_ids).squeeze(-1)
        
        # 4. Compute Score per Sequence in Batch
        # We want to ignore padding (-100) in the average
        mask = (targets.input_ids != -100) & (targets.input_ids != tokenizer.pad_token_id)
        
        for k in range(SEQUENCE_LENGTH):
            # Select valid tokens for the k-th sequence in batch
            valid_log_probs = token_log_probs[k][mask[k]]
            
            if len(valid_log_probs) > 0:
                # Score = Mean Log Probability of the target sequence reconstruction
                score = valid_log_probs.mean().item()
            else:
                score = -100.0
            
            individual_scores.append(score)

    # Media
    avg_score = np.mean(individual_scores)
    # Min
    #min_score = np.min(individual_scores)
    return avg_score, individual_scores

def run_detection_phase(dataset_path, model, tokenizer, device, phase_name, threshold=None):
    print(f"\n--- Starting Phase: {phase_name} ---")
    
    df = pd.read_csv(dataset_path, header=None, names=['payload', 'label'], dtype=str)
    
    texts = df['payload'].tolist()
    labels = df['label'].astype(int).tolist()
    
    final_scores = []
    detailed_scores = [] 
    
    print(f"Processing {len(texts)} sequences...")
    model.eval()
    
    iterator = tqdm(texts, desc=f"Detecting ({phase_name})", unit="seq")
    
    for seq in iterator:
        # Nota: non serve più enumerate(texts) o stampare ogni 100 righe
        avg_score, indiv_scores = evaluate_sequence_leave_one_out(model, tokenizer, seq, device)
        final_scores.append(avg_score)
        detailed_scores.append(indiv_scores)
        
    final_scores = np.array(final_scores)
    
    # --- TEMPORAL SMOOTHING ---
    # Facciamo una media mobile su una finestra di 3 sequenze.
    # Questo riduce il rumore (Falsi Positivi isolati) e evidenzia gli attacchi persistenti.
    # Convertiamo in Pandas Series per comodità
    scores_series = pd.Series(final_scores)
    # Rolling mean (finestra 3). Il 'min_periods=1' serve per non perdere i primi dati.
    #smoothed_scores = scores_series.rolling(window=2, min_periods=1).mean().values
    # span=3 simula una finestra di circa 3, ma con peso esponenziale
    smoothed_scores = scores_series.ewm(span=3, adjust=False).mean().values
    # Sovrascriviamo final_scores con la versione "pulita"
    final_scores = smoothed_scores

    if threshold is None:
        # Threshold calculation on Validation
        calc_threshold = np.percentile(final_scores, 15)
        print(f"Validation Stats -> Mean: {final_scores.mean():.4f}, Min: {final_scores.min():.4f}")
        print(f"Calculated Threshold: {calc_threshold:.4f}")
        return calc_threshold
    else:
        # Prediction on Test
        predictions = (final_scores < threshold).astype(int)
        
        print("\nClassification Report:")
        print(classification_report(labels, predictions, target_names=["Benign", "Attack"]))
        cm = confusion_matrix(labels, predictions)
        print(f"Confusion Matrix:\nTP: {cm[1][1]} | FN: {cm[1][0]}\nFP: {cm[0][1]} | TN: {cm[0][0]}")
        print(f"ROC AUC: {roc_auc_score(labels, -final_scores):.4f}")
        
        # Save Detailed CSV
        results_df = pd.DataFrame({
            'Label': labels,
            'Pred': predictions,
            'Avg_Score': final_scores,
            'Score_P1': [x[0] for x in detailed_scores],
            'Score_P2': [x[1] for x in detailed_scores],
            'Score_P3': [x[2] for x in detailed_scores],
            'Score_P4': [x[3] for x in detailed_scores],
            'Score_P5': [x[4] for x in detailed_scores]
        })
        results_df.to_csv("/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/detection_detailed_results.csv", index=False)
        return threshold

# --- MAIN ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Model from {MODEL_PATH}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
    
    thresh = run_detection_phase(VAL_PATH, model, tokenizer, device, "Validation", threshold=None)
    run_detection_phase(TEST_PATH, model, tokenizer, device, "Test", threshold=thresh)