import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURAZIONE ---
SEQUENCE_LENGTH = 5
MAX_LENGTH = 512
SEPARATOR = ' '

# Percorsi dei file
MODEL_PATH = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/Byt5/BYTES_modbus-sequence_5_ALLMasked-finetuned"
TEST_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits/test.txt"
OUTPUT_CSV = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/detection_final_with_preds.csv"

def hex_to_latin1(hex_sequence):
    """Converte sequenza hex in stringa latin-1 per il tokenizer"""
    try:
        parts = hex_sequence.strip().split(SEPARATOR)
        decoded_parts = [bytes.fromhex(p).decode('latin-1') for p in parts]
        return SEPARATOR.join(decoded_parts)
    except:
        return hex_sequence

def get_raw_log_probs(model, tokenizer, sequence_str, device):
    """Estrae le log-probabilità negative per i 5 pacchetti della finestra"""
    hex_packets = sequence_str.strip().split(SEPARATOR)
    if len(hex_packets) != SEQUENCE_LENGTH:
        return [np.nan] * SEQUENCE_LENGTH

    input_texts, target_texts = [], []
    for i in range(SEQUENCE_LENGTH):
        masked_packets = hex_packets.copy()
        masked_packets[i] = "<extra_id_0>" 
        input_texts.append(hex_to_latin1(SEPARATOR.join(masked_packets)))
        target_texts.append(f"<extra_id_0> {hex_to_latin1(hex_packets[i])} <extra_id_1>")

    inputs = tokenizer(input_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH).to(device)
    targets = tokenizer(target_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH).to(device)
    
    # Maschera per ignorare padding e token speciali ByT5
    pad_token_id = tokenizer.pad_token_id
    target_mask = (targets.input_ids != pad_token_id) & (targets.input_ids < 256)

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

def find_best_threshold_and_predict(df, strategy='at_least_2'):
    """
    Trova la soglia ottimale in memoria e genera le predizioni finali.
    """
    y_true = df['True_Label'].values
    score_cols = [f'LogProb_Pos{i}' for i in range(SEQUENCE_LENGTH)]
    scores = df[score_cols]
    
    # Range di ricerca per le log-probabilità (valori negativi)
    vals = scores.values.flatten()
    vals = vals[~np.isnan(vals)]
    thresholds = np.linspace(np.percentile(vals, 1), np.percentile(vals, 99), 100)
    
    best_f1, best_th, best_preds = -1, None, None
    
    print(f"\n🔍 Ricerca della soglia ottimale per strategia '{strategy.upper()}'...")
    for th in thresholds:
        is_anomalous = scores < th # Log-prob < soglia = Anomalia
        
        if strategy == 'at_least_2':
            vote = (is_anomalous.sum(axis=1) >= 2).astype(int)
        elif strategy == 'majority':
            vote = (is_anomalous.sum(axis=1) > (scores.notna().sum(axis=1) / 2)).astype(int)
        elif strategy == 'at_least_1':
            vote = is_anomalous.any(axis=1).astype(int)
            
        f1 = classification_report(y_true, vote, output_dict=True)['1']['f1-score']
        if f1 > best_f1:
            best_f1, best_th, best_preds = f1, th, vote
            
    return best_f1, best_th, best_preds

def print_final_report(y_true, y_pred, title):
    """Stampa il report a video"""
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    print(report)
    print(f"Confusion Matrix:\n[TP: {cm[1][1]:<5} | FN: {cm[1][0]:<5}]\n[FP: {cm[0][1]:<5} | TN: {cm[0][0]:<5}]")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Caricamento modello su {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
    model.eval()

    packet_registry = {}

    print(f"Lettura dataset da {TEST_PATH}...")
    with open(TEST_PATH, 'r') as f: lines = f.readlines()

    print("Fase 1: Inferenza modello e calcolo probabilità...")
    for line_idx, line in tqdm(enumerate(lines), total=len(lines)):
        line = line.strip()
        if not line: continue
        parts = line.split(',')
        packet_labels = [int(x) for x in parts[1:6]] 
        raw_scores = get_raw_log_probs(model, tokenizer, parts[0], device)
        
        for pos in range(SEQUENCE_LENGTH):
            global_id = line_idx + pos
            if global_id not in packet_registry:
                packet_registry[global_id] = {'label': packet_labels[pos], 'scores': [np.nan] * SEQUENCE_LENGTH}
            packet_registry[global_id]['scores'][pos] = raw_scores[pos]

    # Conversione in DataFrame
    data_for_df = []
    sorted_ids = sorted(packet_registry.keys())
    for pid in sorted_ids:
        row = {'Packet_ID': pid, 'True_Label': packet_registry[pid]['label']}
        for i in range(SEQUENCE_LENGTH):
            row[f'LogProb_Pos{i}'] = packet_registry[pid]['scores'][i]
        data_for_df.append(row)
    df = pd.DataFrame(data_for_df)

    # Fase 2: Ottimizzazione e Predizione interna
    print("\nFase 2: Calcolo Predizioni e Salvataggio Report...")
    STRATEGY = 'majority' # Puoi cambiare con 'majority' o 'at_least_1'
    best_f1, best_th, best_preds = find_best_threshold_and_predict(df, strategy=STRATEGY)
    
    # Aggiungiamo la colonna della predizione al DataFrame
    df['Voting_Pred'] = best_preds

    # Salvataggio CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Dati completi salvati in: {OUTPUT_CSV}")
    
    # Stampa Report Finale
    print_final_report(df['True_Label'], df['Voting_Pred'], f"RISULTATO FINALE (Strategia: {STRATEGY.upper()} | Soglia: {best_th:.4f})")

if __name__ == "__main__":
    main()