import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch.nn.functional as F

# --- CONFIGURAZIONE ---
SEQUENCE_LENGTH = 5
MAX_LENGTH = 512
SEPARATOR = ' '

# Percorsi
MODEL_PATH = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/Byt5/BYTES_modbus-sequence_5_ALLMasked-finetuned"
TEST_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits/test.txt"
OUTPUT_CSV = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/detection_sliding_results.csv"

def hex_to_latin1(hex_sequence):
    """Converte sequenza hex spaziata in stringa latin-1 per il tokenizer"""
    try:
        parts = hex_sequence.strip().split(SEPARATOR)
        decoded_parts = [bytes.fromhex(p).decode('latin-1') for p in parts]
        return SEPARATOR.join(decoded_parts)
    except:
        return hex_sequence

def get_packet_scores(model, tokenizer, sequence_str, device):
    """
    Calcola l'anomaly score basandosi sulla Log-Likelihood negativa.
    Versione FIX: Evita l'uso di indici -100 per il gather su CUDA.
    """
    hex_packets = sequence_str.strip().split(SEPARATOR)
    if len(hex_packets) != SEQUENCE_LENGTH:
        return [np.nan] * SEQUENCE_LENGTH

    input_texts = []
    target_texts = []
    
    for i in range(SEQUENCE_LENGTH):
        masked_packets = hex_packets.copy()
        
        # Maschera il pacchetto corrente
        masked_packets[i] = "<extra_id_0>" 
        
        input_text = hex_to_latin1(SEPARATOR.join(masked_packets))
        
        # Target: sentinel + contenuto + sentinel finale
        target_content = hex_to_latin1(hex_packets[i])
        target_text = f"<extra_id_0> {target_content} <extra_id_1>"
        
        input_texts.append(input_text)
        target_texts.append(target_text)

    # Tokenizzazione
    inputs = tokenizer(input_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH).to(device)
    targets = tokenizer(target_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH).to(device)
    # Teniamo i token originali (es. 0 per padding) per fare il gather senza errori.
    # Creiamo solo una maschera booleana per sapere cosa ignorare dopo.
    pad_token_id = tokenizer.pad_token_id
    target_mask = (targets.input_ids != pad_token_id)

    individual_scores = []

    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=targets.input_ids)
        
        logits = outputs.logits 
        log_probs = F.log_softmax(logits, dim=-1) # (Batch, Seq_Len, Vocab)
        
        # Usa gli ID originali (compresi i padding) per estrarre le probabilità
        target_ids = targets.input_ids.unsqueeze(-1) # (Batch, Seq_Len, 1)
        
        # Gather ora è sicuro perché non ci sono -100
        token_log_probs = log_probs.gather(-1, target_ids).squeeze(-1) # (Batch, Seq_Len)
        
        for k in range(SEQUENCE_LENGTH):
            # Ora applichiamo la maschera per prendere solo i token VERI
            # target_mask[k] è False dove c'era padding
            valid_log_probs = token_log_probs[k][target_mask[k]]
            
            if len(valid_log_probs) > 0:
                mean_log_prob = valid_log_probs.mean().item()
                # Inverti il segno: Più è basso il log-prob (es -5), più alto è lo score (5)
                score = -1 * mean_log_prob
            else:
                score = np.nan
            
            individual_scores.append(score)
            
    return individual_scores

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Caricamento modello da {MODEL_PATH} su {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
    model.eval()

    # Dizionario per accumulare i dati dei pacchetti
    # Chiave: Global_Packet_ID (int)
    # Valore: { 'label': int, 'scores': {0: s, 1: s, ...} }
    packet_registry = {}

    print(f"Lettura dataset da {TEST_PATH}...")
    with open(TEST_PATH, 'r') as f:
        lines = f.readlines()

    print("Inizio Detection con Sliding Window Accumulation...")
    
    for line_idx, line in tqdm(enumerate(lines), total=len(lines)):
        line = line.strip()
        if not line: continue
        
        # Parsing: <sequenza>, <l1>, <l2>, <l3>, <l4>, <l5>, <l_seq>
        parts = line.split(',')
        sequence_str = parts[0]
        # Le label dei singoli pacchetti sono agli indici 1, 2, 3, 4, 5
        packet_labels = [int(x) for x in parts[1:6]] 
        
        # Calcola scores per i 5 pacchetti di questa finestra
        current_scores = get_packet_scores(model, tokenizer, sequence_str, device)
        
        # Assegna i risultati ai pacchetti globali corretti
        for pos in range(SEQUENCE_LENGTH):
            # FORMULA CHIAVE: Global ID = Indice Riga + Posizione Relativa
            global_id = line_idx + pos
            
            if global_id not in packet_registry:
                packet_registry[global_id] = {
                    'label': packet_labels[pos],
                    'scores': [np.nan] * SEQUENCE_LENGTH # Inizializza con NaN
                }
            
            # Salva lo score ottenuto trovandosi in posizione 'pos'
            packet_registry[global_id]['scores'][pos] = current_scores[pos]

    # --- Creazione DataFrame Finale ---
    print("Creazione CSV aggregato...")
    data_for_df = []
    
    # Ordiniamo per ID per mantenere l'ordine temporale
    sorted_ids = sorted(packet_registry.keys())
    
    for pid in sorted_ids:
        entry = packet_registry[pid]
        row = {
            'Packet_ID': pid,
            'True_Label': entry['label'],
            'Score_Pos0': entry['scores'][0],
            'Score_Pos1': entry['scores'][1],
            'Score_Pos2': entry['scores'][2],
            'Score_Pos3': entry['scores'][3],
            'Score_Pos4': entry['scores'][4]
        }
        data_for_df.append(row)

    df_results = pd.DataFrame(data_for_df)
    
    # Salvataggio
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Analisi completata. Risultati salvati in: {OUTPUT_CSV}")
    print(f"   Totale pacchetti unici analizzati: {len(df_results)}")

if __name__ == "__main__":
    main()