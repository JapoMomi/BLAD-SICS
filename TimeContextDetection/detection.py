import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score

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
        parts = hex_sequence.strip().split(SEPARATOR)
        decoded_parts = [bytes.fromhex(p).decode('latin-1') for p in parts]
        return SEPARATOR.join(decoded_parts)
    except:
        return hex_sequence

def evaluate_sequence_leave_one_out(model, tokenizer, sequence_str, device):
    # 1. Split sicuro sulla stringa HEX
    hex_packets = sequence_str.strip().split(SEPARATOR)
    packets = []
    for h_pack in hex_packets:
        try:
            dec_p = bytes.fromhex(h_pack).decode('latin-1')
            packets.append(dec_p)
        except ValueError:
            packets.append(h_pack)
    
    if len(packets) != SEQUENCE_LENGTH:
        return -100.0, -100.0, [-100.0] * SEQUENCE_LENGTH

    individual_scores = []
    input_texts = []
    target_texts = []
    
    for i in range(SEQUENCE_LENGTH):
        masked_packets = packets.copy()
        masked_packets[i] = "<extra_id_0>"
        input_text = SEPARATOR.join(masked_packets)
        target_text = f"<extra_id_0> {packets[i]} <extra_id_1>"
        input_texts.append(input_text)
        target_texts.append(target_text)
        
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
    targets = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)

    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=targets.input_ids)
        logits = outputs.logits 
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        target_ids = targets.input_ids.unsqueeze(-1)
        token_log_probs = log_probs.gather(-1, target_ids).squeeze(-1)
        
        mask = (targets.input_ids != -100) & (targets.input_ids != tokenizer.pad_token_id)
        
        for k in range(SEQUENCE_LENGTH):
            valid_log_probs = token_log_probs[k][mask[k]]
            if len(valid_log_probs) > 0:
                score = valid_log_probs.mean().item()
            else:
                score = -100.0
            individual_scores.append(score)

    avg_score = np.mean(individual_scores)
    min_score = np.min(individual_scores) # Calcoliamo anche il minimo qui
    return avg_score, min_score, individual_scores

def find_best_hybrid_thresholds(labels, avg_scores, min_scores):
    """
    Esegue una Grid Search per trovare la combinazione migliore di Avg_Threshold e Min_Threshold
    che massimizza l'F1-Score sul Validation Set.
    """
    print("\n🔍 Avvio Grid Search per le soglie ottimali...")
    
    # Definiamo i candidati usando i percentili per adattarci alla distribuzione
    # Testiamo 50 valori possibili per la media e 50 per il minimo
    avg_candidates = np.unique(np.percentile(avg_scores, np.linspace(0.5, 30, 50)))
    min_candidates = np.unique(np.percentile(min_scores, np.linspace(0.1, 20, 50)))
    
    best_f1 = -1
    best_avg = 0
    best_min = 0
    
    # Grid Search
    for t_avg in avg_candidates:
        for t_min in min_candidates:
            # Deve essere SOLO (Min < t_min) che è più severo, quindi t_min deve essere <= t_avg
            # Ma lasciamo libertà al search
            
            # Logica Ibrida: Allarme se (Avg < T_avg) OR (Min < T_min)
            preds = ((avg_scores < t_avg) | (min_scores < t_min)).astype(int)
            
            # Calcolo F1 Score (usiamo 'macro' o binary per la classe 1)
            current_f1 = f1_score(labels, preds, pos_label=1)
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_avg = t_avg
                best_min = t_min
                
    print(f"✅ Trovate soglie ottimali (F1 Validation: {best_f1:.4f})")
    print(f"   -> Avg Threshold: {best_avg:.4f}")
    print(f"   -> Min Threshold: {best_min:.4f}")
    
    return best_avg, best_min

def run_detection_phase(dataset_path, model, tokenizer, device, phase_name, thresholds=None):
    print(f"\n--- Starting Phase: {phase_name} ---")
    
    df = pd.read_csv(dataset_path, header=None, names=['payload', 'label'], dtype=str)
    texts = df['payload'].tolist()
    labels = df['label'].astype(int).tolist()
    
    avg_scores_list = []
    min_scores_list = []
    detailed_scores_list = []
    
    print(f"Processing {len(texts)} sequences...")
    model.eval()
    
    iterator = tqdm(texts, desc=f"Detecting ({phase_name})", unit="seq")
    
    for seq in iterator:
        avg_s, min_s, indiv_s = evaluate_sequence_leave_one_out(model, tokenizer, seq, device)
        avg_scores_list.append(avg_s)
        min_scores_list.append(min_s)
        detailed_scores_list.append(indiv_s)
        
    # Convertiamo in numpy array per velocità
    final_avg_scores = np.array(avg_scores_list)
    final_min_scores = np.array(min_scores_list)
    labels = np.array(labels)

    # --- TEMPORAL SMOOTHING (Solo sulla Media, il Minimo è impulsivo) ---
    scores_series = pd.Series(final_avg_scores)
    smoothed_avg_scores = scores_series.ewm(span=3, adjust=False).mean().values
    final_avg_scores = smoothed_avg_scores

    # --- CALCOLO O APPLICAZIONE SOGLIE ---
    if thresholds is None:
        # Siamo in VALIDATION: Dobbiamo calcolare le soglie
        best_avg, best_min = find_best_hybrid_thresholds(labels, final_avg_scores, final_min_scores)
        return (best_avg, best_min)
    else:
        # Siamo in TEST: Usiamo le soglie passate
        thresh_avg, thresh_min = thresholds
        
        # --- APPLICAZIONE LOGICA IBRIDA ---
        # Condizione A: Media brutta (Attacco persistente)
        cond_A = (final_avg_scores < thresh_avg)
        
        # Condizione B: Minimo orribile (Attacco puntuale/cecchino)
        cond_B = (final_min_scores < thresh_min)
        
        # OR Logico
        raw_predictions = (cond_A | cond_B).astype(int)
        
        # --- PERSISTENZA (Opzionale, attivata) ---
        persistence_window = 3
        filtered_predictions = np.zeros_like(raw_predictions)
        kernel = np.ones(persistence_window)
        # Convoluzione: somma mobile. Se somma == 3, allora 3 '1' consecutivi.
        conv_result = np.convolve(raw_predictions, kernel, mode='valid')
        # Padding iniziale
        padding = np.zeros(persistence_window - 1)
        detected_indices = (conv_result == persistence_window).astype(int)
        filtered_predictions = np.concatenate([padding, detected_indices])
        
        final_predictions = filtered_predictions
        
        print("\nClassification Report (Hybrid + Persistence):")
        print(classification_report(labels, final_predictions, target_names=["Benign", "Attack"]))
        cm = confusion_matrix(labels, final_predictions)
        print(f"Confusion Matrix:\nTP: {cm[1][1]} | FN: {cm[1][0]}\nFP: {cm[0][1]} | TN: {cm[0][0]}")
        
        # Salvataggio CSV Dettagliato
        results_df = pd.DataFrame({
            'Label': labels,
            'Pred': final_predictions,
            'Avg_Score': final_avg_scores,
            'Min_Score': final_min_scores, # Aggiunto colonna Min_Score
            'Score_P1': [x[0] for x in detailed_scores_list],
            'Score_P2': [x[1] for x in detailed_scores_list],
            'Score_P3': [x[2] for x in detailed_scores_list],
            'Score_P4': [x[3] for x in detailed_scores_list],
            'Score_P5': [x[4] for x in detailed_scores_list]
        })
        results_df.to_csv("/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/detection_detailed_results.csv", index=False)
        return thresholds

# --- MAIN ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Model from {MODEL_PATH}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
    
    # 1. Calcola soglie ottimali su VALIDATION
    best_thresholds = run_detection_phase(VAL_PATH, model, tokenizer, device, "Validation", thresholds=None)
    
    # 2. Applica le soglie su TEST
    run_detection_phase(TEST_PATH, model, tokenizer, device, "Test", thresholds=best_thresholds)