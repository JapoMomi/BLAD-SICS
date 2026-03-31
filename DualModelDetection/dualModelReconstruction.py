import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

# --- CONFIGURAZIONE ---
SEQUENCE_LENGTH = 5
MAX_LENGTH = 512
SEPARATOR = ' '
MASK_TOKEN_ID = 258 # <extra_id_0> per ByT5 [cite: 2026-01-07]

# Percorsi Modelli
PATH_SINGLE = "/home/spritz/storage/disk0/Master_Thesis/SingplePacketDetection/Byt5/BYTES_modbus-single_packet-finetuned"
PATH_CONTEXT = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/Byt5/BYTES_modbus-sequence_5_ALLMasked-finetuned"

# Percorsi Dataset Input
VAL_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits/validation.txt"
TEST_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits/test.txt"

# Percorsi Output CSV
OUTPUT_VAL_CSV = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_validation_results.csv"
OUTPUT_TEST_CSV = "/home/spritz/storage/disk0/Master_Thesis/DualApprachDetection/dual_model_detection_results.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FUNZIONI DI UTILITÀ ---

def hex_to_latin1(hex_str):
    """Converte esadecimale in latin-1 raw bytes."""
    try:
        return bytes.fromhex(hex_str).decode('latin-1')
    except:
        return ""

def get_single_packet_log_prob(model, tokenizer, packet_hex, window_size=5):
    """
    Calcola lo score sintattico (SinglePacket) usando sliding window masking sui byte.
    Eseguito UNA SOLA VOLTA per pacchetto univoco.
    Restituisce sia la MEDIA che il MINIMO della finestra scorrevole.
    """
    packet_latin1 = hex_to_latin1(packet_hex)
    inputs = tokenizer(packet_latin1, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids
    
    # Rimuovi EOS se presente per la scansione
    if input_ids[0, -1] == 1: 
        input_ids = input_ids[:, :-1]
        
    seq_len = input_ids.shape[1]
    if seq_len <= window_size: 
        return 0.0, 0.0 # Ritorna tupla (mean, min)

    chunk_scores = []
    # Slide pixel-by-pixel (stride=1) per massima precisione
    for j in range(0, seq_len - window_size + 1):
        masked_input = input_ids.clone()
        masked_input[0, j:j+window_size] = MASK_TOKEN_ID
        
        with torch.no_grad():
            outputs = model(input_ids=masked_input, labels=input_ids)
            log_probs = F.log_softmax(outputs.logits, dim=-1)
            
            target_ids = input_ids.unsqueeze(-1)
            token_log_probs = log_probs.gather(-1, target_ids).squeeze(-1)
            
            # Media log-prob nella finestra mascherata
            window_score = token_log_probs[0, j:j+window_size].mean().item()
            chunk_scores.append(window_score)
            
    if chunk_scores:
        return np.mean(chunk_scores), np.min(chunk_scores)
    else:
        return 0.0, 0.0

def get_context_log_probs(model, tokenizer, sequence_str):
    """Calcola le 5 log-probabilità per i pacchetti nel loro contesto (modello Context)."""
    hex_packets = sequence_str.strip().split(SEPARATOR)
    if len(hex_packets) != SEQUENCE_LENGTH:
        return [np.nan] * SEQUENCE_LENGTH

    # Decodifica tutti i pacchetti da esadecimale a Latin-1 in anticipo
    latin1_packets = [hex_to_latin1(hp) for hp in hex_packets]

    input_texts = []
    target_texts = []
    
    for i in range(SEQUENCE_LENGTH):
        # Lavoriamo sulla lista di byte già decodificati
        masked_packets = latin1_packets.copy()
        masked_packets[i] = "<extra_id_0>"
        
        input_texts.append(SEPARATOR.join(masked_packets))
        target_texts.append(f"<extra_id_0> {latin1_packets[i]} <extra_id_1>")

    inputs = tokenizer(input_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    targets = tokenizer(target_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    
    target_mask = (targets.input_ids != tokenizer.pad_token_id) & (targets.input_ids < 256)

    scores = []
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=targets.input_ids)
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        
        target_ids = targets.input_ids.unsqueeze(-1)
        token_log_probs = log_probs.gather(-1, target_ids).squeeze(-1)
        
        for k in range(SEQUENCE_LENGTH):
            valid_log_probs = token_log_probs[k][target_mask[k]]
            scores.append(valid_log_probs.mean().item() if len(valid_log_probs) > 0 else np.nan)
            
    return scores

def process_dataset(filepath, model_single, tok_single, model_context, tok_context, device, desc="Processing"):
    """
    Legge il file di dataset, calcola gli score (Single + Context) 
    e organizza i dati a livello di singolo pacchetto (Packet-Level).
    Restituisce un DataFrame pandas.
    """
    print(f"\nLettura dataset da {filepath}...")
    with open(filepath, 'r') as f:
        lines = f.readlines()

    packet_registry = {}

    for line_idx, line in tqdm(enumerate(lines), total=len(lines), desc=desc):
        line = line.strip()
        if not line: continue
        
        parts = line.split(',')
        payload_seq = parts[0]
        packet_labels = [int(x) for x in parts[1:6]]
        
        # 1. Calcolo Score Contestuali (5 score)
        ctx_scores = get_context_log_probs(model_context, tok_context, payload_seq)
        
        # 2. Assegnazione ai singoli pacchetti globali
        hex_packets = payload_seq.split(SEPARATOR)
        
        for pos in range(SEQUENCE_LENGTH):
            global_id = line_idx + pos
            
            # Inizializzazione pacchetto nel registry
            if global_id not in packet_registry:
                packet_registry[global_id] = {
                    'label': packet_labels[pos], 
                    'single_score': np.nan, 
                    'min_single_score': np.nan, # Nuova chiave per il minimo
                    'ctx_scores': [np.nan] * SEQUENCE_LENGTH
                }
                
            # Salva lo score contestuale nella posizione relativa (0-4)
            packet_registry[global_id]['ctx_scores'][pos] = ctx_scores[pos]
            
            # Calcolo Single Score (lo fa solo una volta per pacchetto)
            if np.isnan(packet_registry[global_id]['single_score']):
                target_hex = hex_packets[pos]
                # Ora spacchettiamo i due valori restituiti
                s_mean, s_min = get_single_packet_log_prob(model_single, tok_single, target_hex)
                packet_registry[global_id]['single_score'] = s_mean
                packet_registry[global_id]['min_single_score'] = s_min

    # Creazione DataFrame Finale
    print(f"[{desc}] Creazione DataFrame...")
    data = []
    sorted_ids = sorted(packet_registry.keys())
    
    for pid in sorted_ids:
        entry = packet_registry[pid]
        row = {
            'Packet_ID': pid,
            'True_Label': entry['label'],
            'Single_Score': entry['single_score'],
            'Min_Single_Score': entry['min_single_score'] # Aggiungiamo la colonna al row dict
        }
        for i in range(SEQUENCE_LENGTH):
            row[f'Ctx_Pos{i}'] = entry['ctx_scores'][i]
        data.append(row)
        
    df = pd.DataFrame(data)
    return df

# --- MAIN FLOW ---

def main():
    print(f"Inizializzazione ambiente su {DEVICE}...")
    
    # Caricamento Modello SINGLE
    print("\nCaricamento Modello SINGLE...")
    tok_single = AutoTokenizer.from_pretrained(PATH_SINGLE, local_files_only=True)
    model_single = T5ForConditionalGeneration.from_pretrained(PATH_SINGLE, local_files_only=True).to(DEVICE)
    model_single.eval()

    # Caricamento Modello CONTEXT
    print("\nCaricamento Modello CONTEXT...")
    tok_context = AutoTokenizer.from_pretrained(PATH_CONTEXT, local_files_only=True)
    model_context = T5ForConditionalGeneration.from_pretrained(PATH_CONTEXT, local_files_only=True).to(DEVICE)
    model_context.eval()

    # --- ELABORAZIONE VALIDATION SET ---
    print("\n" + "="*50)
    print(" FASE 1: Estrazione Score per VALIDATION SET")
    print("="*50)
    df_val = process_dataset(VAL_PATH, model_single, tok_single, model_context, tok_context, DEVICE, desc="Validation")
    
    print(f"Salvataggio Validation in: {OUTPUT_VAL_CSV}")
    df_val.to_csv(OUTPUT_VAL_CSV, index=False)
    
    # --- ELABORAZIONE TEST SET ---
    print("\n" + "="*50)
    print(" FASE 2: Estrazione Score per TEST SET")
    print("="*50)
    df_test = process_dataset(TEST_PATH, model_single, tok_single, model_context, tok_context, DEVICE, desc="Test")
    
    print(f"Salvataggio Test in: {OUTPUT_TEST_CSV}")
    df_test.to_csv(OUTPUT_TEST_CSV, index=False)

    print("\n✅ Estrazione completata con successo! I dati sono pronti per l'Anomaly Detection Unsupervised.")

if __name__ == "__main__":
    main()