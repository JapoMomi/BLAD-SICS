import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch.nn.functional as F

# --- CONFIGURAZIONE ---
SEQUENCE_LENGTH = 5  # Numero di pacchetti nella sliding window
MAX_LENGTH = 512     # Lunghezza massima in token per l'input del modello
SEPARATOR = ' '      # Separatore usato nel file di testo tra i pacchetti hex

# Percorsi dei file (da aggiornare con i tuoi path)
MODEL_PATH = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/Byt5/BYTES_modbus-sequence_5_ALLMasked-finetuned"
TEST_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits/test.txt"
OUTPUT_CSV = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/detection_raw_logprobs.csv"

def hex_to_latin1(hex_sequence):
    """
    I modelli byte-level come ByT5 lavorano sui byte grezzi. 
    Questa funzione converte la stringa esadecimale (es. "0403") nei byte reali 
    rappresentati come stringa latin-1, che è il formato richiesto dal tokenizer.
    """
    try:
        parts = hex_sequence.strip().split(SEPARATOR)
        decoded_parts = [bytes.fromhex(p).decode('latin-1') for p in parts]
        return SEPARATOR.join(decoded_parts)
    except:
        return hex_sequence

def get_raw_log_probs(model, tokenizer, sequence_str, device):
    """
    Riceve una sequenza di 5 pacchetti e restituisce 5 probabilità (una per pacchetto).
    Ogni probabilità indica quanto il modello si aspettava quel pacchetto.
    Più il valore è vicino a 0, più il pacchetto è normale.
    Più il valore è negativo (es. -5.0), più il pacchetto è anomalo.
    """
    hex_packets = sequence_str.strip().split(SEPARATOR)
    if len(hex_packets) != SEQUENCE_LENGTH:
        return [np.nan] * SEQUENCE_LENGTH # Ritorna NaN se la riga è corrotta

    input_texts = []
    target_texts = []
    
    # Prepariamo 5 test diversi (uno per ogni posizione della finestra)
    for i in range(SEQUENCE_LENGTH):
        masked_packets = hex_packets.copy()
        # Mascheriamo il pacchetto i-esimo (è il pacchetto che il modello dovrà indovinare)
        masked_packets[i] = "<extra_id_0>" 
        input_text = hex_to_latin1(SEPARATOR.join(masked_packets))
        
        # Il target è il pacchetto VERO circondato dalle sentinelle
        target_content = hex_to_latin1(hex_packets[i])
        target_text = f"<extra_id_0> {target_content} <extra_id_1>"
        
        input_texts.append(input_text)
        target_texts.append(target_text)

    # Convertiamo il testo in tensori (numeri) per la GPU
    inputs = tokenizer(input_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH).to(device)
    targets = tokenizer(target_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH).to(device)
    
    # Creiamo una maschera per ignorare il "rumore" nel calcolo della probabilità:
    # 1. Ignoriamo il padding (spazi vuoti alla fine)
    # 2. Ignoriamo i token speciali di ByT5 (che hanno ID >= 256)
    pad_token_id = tokenizer.pad_token_id
    target_mask = (targets.input_ids != pad_token_id) & (targets.input_ids < 256)

    scores = []

    # Disabilitiamo il calcolo dei gradienti per risparmiare memoria (siamo in inferenza)
    with torch.no_grad():
        # Diamo l'input al modello e otteniamo i logits (i punteggi grezzi di predizione)
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=targets.input_ids)
        logits = outputs.logits 
        
        # Trasformiamo i logits in log-probabilità (numeri negativi che sommano a 1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Estraiamo solo le probabilità relative ai byte corretti (quelli del target)
        target_ids = targets.input_ids.unsqueeze(-1)
        token_log_probs = log_probs.gather(-1, target_ids).squeeze(-1)
        
        # Calcoliamo la media per ognuno dei 5 pacchetti testati
        for k in range(SEQUENCE_LENGTH):
            # Usiamo la maschera per prendere solo i byte reali (escludendo padding/token speciali)
            valid_log_probs = token_log_probs[k][target_mask[k]]
            
            if len(valid_log_probs) > 0:
                # Media pura: se è -0.1 il pacchetto è normale, se è -4.5 è anomalo
                mean_log_prob = valid_log_probs.mean().item()
            else:
                mean_log_prob = np.nan
            
            scores.append(mean_log_prob)
            
    return scores

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Caricamento modello su {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
    model.eval() # Imposta il modello in modalità valutazione

    # Dizionario che raccoglierà i dati di un pacchetto da tutte le finestre in cui appare
    packet_registry = {}

    print(f"Lettura dataset da {TEST_PATH}...")
    with open(TEST_PATH, 'r') as f:
        lines = f.readlines()

    print("Estrazione probabilità raw con Sliding Window...")
    # Iteriamo su ogni riga del file di test
    for line_idx, line in tqdm(enumerate(lines), total=len(lines)):
        line = line.strip()
        if not line: continue
        
        # Leggiamo i dati: la sequenza e le 5 etichette (1 per pacchetto)
        parts = line.split(',')
        sequence_str = parts[0]
        packet_labels = [int(x) for x in parts[1:6]] 
        
        # Otteniamo le probabilità dei 5 pacchetti di questa riga
        raw_scores = get_raw_log_probs(model, tokenizer, sequence_str, device)
        
        # --- PARTE CHIAVE: Ricostruzione Temporale ---
        # Salviamo il punteggio nella posizione corretta per il pacchetto globale
        for pos in range(SEQUENCE_LENGTH):
            # global_id identifica UNIVOCAMENTE il pacchetto nell'intero dataset
            global_id = line_idx + pos
            
            # Se è la prima volta che vediamo questo pacchetto, creiamo il suo spazio
            if global_id not in packet_registry:
                packet_registry[global_id] = {
                    'label': packet_labels[pos],
                    'scores': [np.nan] * SEQUENCE_LENGTH # Inizializza 5 spazi vuoti
                }
            
            # Salviamo il punteggio ottenuto quando il pacchetto era in posizione 'pos'
            packet_registry[global_id]['scores'][pos] = raw_scores[pos]

    # --- CREAZIONE DEL CSV FINALE ---
    print("Creazione CSV finale...")
    data_for_df = []
    # Ordiniamo per ID per mantenere l'ordine cronologico del traffico di rete
    sorted_ids = sorted(packet_registry.keys())
    
    for pid in sorted_ids:
        entry = packet_registry[pid]
        row = {'Packet_ID': pid, 'True_Label': entry['label']}
        # Creiamo le colonne per i punteggi nelle 5 posizioni (Pos0, Pos1... Pos4)
        for i in range(SEQUENCE_LENGTH):
            row[f'LogProb_Pos{i}'] = entry['scores'][i]
        data_for_df.append(row)

    df_results = pd.DataFrame(data_for_df)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Analisi completata. Salvato in: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()