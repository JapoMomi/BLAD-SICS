import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURAZIONE ---
SEQUENCE_LENGTH = 5
MAX_LENGTH = 512
SEPARATOR = ' '
MASK_TOKEN_ID = 258 # <extra_id_0> per ByT5

# Percorsi
PATH_SINGLE = "/home/spritz/storage/disk0/Master_Thesis/SingplePacketDetection/Byt5/BYTES_modbus-single_packet-finetuned"
PATH_CONTEXT = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/Byt5/BYTES_modbus-sequence_5_ALLMasked-finetuned"
TEST_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits/test.txt"
OUTPUT_CSV = "dual_model_detection_results.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FUNZIONI DI UTILITÀ ---

def hex_to_latin1(hex_str):
    """Converte esadecimale in latin-1 raw bytes."""
    try:
        return bytes.fromhex(hex_str).decode('latin-1')
    except:
        return ""

def get_single_packet_score(model, tokenizer, packet_hex, window_size=5):
    """
    Calcola lo score sintattico (SinglePacket) usando sliding window masking sui byte.
    Eseguito UNA SOLA VOLTA per pacchetto univoco.
    """
    packet_latin1 = hex_to_latin1(packet_hex)
    inputs = tokenizer(packet_latin1, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids
    
    # Rimuovi EOS se presente per la scansione
    if input_ids[0, -1] == 1: 
        input_ids = input_ids[:, :-1]
        
    seq_len = input_ids.shape[1]
    if seq_len <= window_size: return 0.0

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
            
    return np.mean(chunk_scores) if chunk_scores else 0.0

def get_context_scores_batch(model, tokenizer, sequence_hex_str):
    """
    Calcola 5 score contestuali (uno per ogni posizione nella sequenza).
    Logica identica al tuo script: maschera P_i, predici P_i.
    """
    hex_packets = sequence_hex_str.strip().split(SEPARATOR)
    if len(hex_packets) != SEQUENCE_LENGTH:
        return [np.nan] * SEQUENCE_LENGTH

    input_texts, target_texts = [], []
    
    # Prepariamo i 5 input mascherati (Mask P0, Mask P1, ..., Mask P4)
    for i in range(SEQUENCE_LENGTH):
        masked_packets = hex_packets.copy()
        masked_packets[i] = "<extra_id_0>" # Masking token placeholder
        
        # Input: P0 P1 <extra_id_0> P3 P4
        input_texts.append(hex_to_latin1(SEPARATOR.join(masked_packets)))
        
        # Target: <extra_id_0> P2 <extra_id_1>
        target_texts.append(f"<extra_id_0> {hex_to_latin1(hex_packets[i])} <extra_id_1>")

    # Tokenizzazione Batch
    inputs = tokenizer(input_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    targets = tokenizer(target_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    
    scores = []
    pad_token_id = tokenizer.pad_token_id
    
    # Ignoriamo padding e token speciali nel calcolo della media
    target_mask = (targets.input_ids != pad_token_id) & (targets.input_ids < 256)

    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=targets.input_ids)
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        
        target_ids = targets.input_ids.unsqueeze(-1)
        token_log_probs = log_probs.gather(-1, target_ids).squeeze(-1)
        
        for k in range(SEQUENCE_LENGTH):
            valid_log_probs = token_log_probs[k][target_mask[k]]
            if len(valid_log_probs) > 0:
                scores.append(valid_log_probs.mean().item())
            else:
                scores.append(np.nan)
            
    return scores # Ritorna lista di 5 float

# --- LOGICA DI VOTING ---
def find_best_threshold_dual(df):
    """
    Trova la soglia ottimale combinando SingleScore e MeanContextScore.
    """
    y_true = df['True_Label'].values
    
    # Calcoliamo lo score contestuale medio (ignorando i NaN dove la finestra non copriva)
    context_cols = [f'Ctx_Pos{i}' for i in range(SEQUENCE_LENGTH)]
    df['Avg_Context'] = df[context_cols].mean(axis=1)
    
    # Creiamo uno Score Combinato (Somma pesata o semplice media)
    # Poiché sono Log-Probs (negativi), sommarli è come moltiplicare le probabilità.
    # Normalizziamo grossolanamente: SingleScore tende ad essere più alto (meno negativo).
    df['Final_Score'] = df['Single_Score'] + df['Avg_Context']
    
    scores = df['Final_Score'].values
    scores = scores[~np.isnan(scores)]
    
    # Cerchiamo la soglia
    thresholds = np.linspace(np.percentile(scores, 1), np.percentile(scores, 99), 100)
    best_f1, best_th, best_preds = -1, None, None
    
    print("\n🔍 Ricerca soglia ottimale su (Single + Context)...")
    for th in thresholds:
        preds = (df['Final_Score'] < th).astype(int)
        rep = classification_report(y_true, preds, output_dict=True, zero_division=0)
        f1 = rep['1']['f1-score']
        
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_preds = preds
            
    return best_f1, best_th, best_preds

# --- MAIN ---
if __name__ == "__main__":
    print(f"Caricamento modelli su {DEVICE}...")
    
    # 1. Carica Single Packet Model
    tok_single = AutoTokenizer.from_pretrained(PATH_SINGLE, local_files_only=True)
    mod_single = T5ForConditionalGeneration.from_pretrained(PATH_SINGLE, local_files_only=True).to(DEVICE).eval()
    
    # 2. Carica Time Context Model
    tok_context = AutoTokenizer.from_pretrained(PATH_CONTEXT, local_files_only=True) # Usa tokenizer contestuale se diverso, altrimenti ByT5 base è uguale
    mod_context = T5ForConditionalGeneration.from_pretrained(PATH_CONTEXT, local_files_only=True).to(DEVICE).eval()

    # Registry: Key = Packet Global ID
    packet_registry = {}

    print(f"Lettura {TEST_PATH}...")
    with open(TEST_PATH, 'r') as f: lines = f.readlines()

    print("Inizio scansione sequenze...")
    # Usiamo tqdm per progress bar
    for line_idx, line in tqdm(enumerate(lines), total=len(lines)):
        line = line.strip()
        if not line: continue
        
        parts = line.split(',')
        seq_hex_str = parts[0]
        hex_packets = seq_hex_str.split(SEPARATOR)
        
        # Labels dei 5 pacchetti nella sequenza corrente
        labels_in_window = [int(x) for x in parts[1:6]]
        
        # 1. Calcola i 5 Context Scores (uno per posizione)
        ctx_scores = get_context_scores_batch(mod_context, tok_context, seq_hex_str)
        
        # 2. Aggiorna il registry
        for pos in range(SEQUENCE_LENGTH):
            # Calcolo ID univoco del pacchetto nel flusso
            # Assumendo sliding window di step 1: il pacchetto 0 della riga 1 è lo stesso della riga 0 pos 1?
            # Se il dataset è generato da splitter.py con sliding window:
            # Riga 0: P0, P1, P2, P3, P4
            # Riga 1: P1, P2, P3, P4, P5
            # Quindi Global_ID = line_idx + pos
            global_id = line_idx + pos
            
            if global_id not in packet_registry:
                packet_registry[global_id] = {
                    'hex': hex_packets[pos],
                    'label': labels_in_window[pos],
                    'single_score': None,      # Da calcolare
                    'ctx_scores': [np.nan] * SEQUENCE_LENGTH # [Score quando era in Pos0, Score quando era in Pos1...]
                }
            
            # Salva lo score contestuale nella posizione relativa corretta
            packet_registry[global_id]['ctx_scores'][pos] = ctx_scores[pos]
            
            # 3. Calcola Single Score (SOLO SE MANCANTE - Ottimizzazione)
            if packet_registry[global_id]['single_score'] is None:
                s_score = get_single_packet_score(mod_single, tok_single, hex_packets[pos])
                packet_registry[global_id]['single_score'] = s_score

    # --- Creazione DataFrame Finale ---
    print("Creazione DataFrame...")
    data = []
    sorted_ids = sorted(packet_registry.keys())
    
    for pid in sorted_ids:
        entry = packet_registry[pid]
        row = {
            'Packet_ID': pid,
            'True_Label': entry['label'],
            'Single_Score': entry['single_score']
        }
        # Aggiungi colonne Ctx_Pos0 ... Ctx_Pos4
        for i in range(SEQUENCE_LENGTH):
            row[f'Ctx_Pos{i}'] = entry['ctx_scores'][i]
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # --- Analisi e Prediction ---
    f1, th, preds = find_best_threshold_dual(df)
    
    df['Pred_Label'] = preds
    
    print("\n" + "="*60)
    print(f"REPORT DUAL MODEL (Single + Context)")
    print(f"Best Threshold (Sum LogProbs): {th:.4f}")
    print("="*60)
    print(classification_report(df['True_Label'], df['Pred_Label'], digits=4))
    
    cm = confusion_matrix(df['True_Label'], df['Pred_Label'])
    print(f"Confusion Matrix:\n[TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}]\n[FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}]")
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Salvati risultati dettagliati in {OUTPUT_CSV}")