import torch
from transformers import AutoTokenizer

# --- CONFIGURAZIONE ---
PATH_CONTEXT = "/home/spritz/storage/disk0/Master_Thesis/TimeContextDetection/Byt5/BYTES_modbus-sequence_5_ALLMasked-finetuned"
SEQUENCE_LENGTH = 5
SEPARATOR = ' '

def hex_to_latin1(hex_str):
    try:
        return bytes.fromhex(hex_str).decode('latin-1')
    except:
        return ""

def main():
    print(f"Caricamento tokenizer da: {PATH_CONTEXT}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(PATH_CONTEXT, local_files_only=True)
    except Exception as e:
        print(f"Errore nel caricamento locale, fallback su base model... ({e})")
        tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    # Sequenza di esempio
    sequence_str = "04100be9001224000000000001000000000000400a3d71700000003e8ca57a000000003f1eb8523fa147ae3334 04100be900124536 04030bb700093245 04031210000e0011f00000000000000000418658473632 04100be9001224000000000001000000000000400a3d71700000003e8ca57a000000003f1eb8523fa147ae3334"
    hex_packets = sequence_str.strip().split(SEPARATOR)

    print("\n" + "="*60)
    print(" FASE 1 CORRETTA: DECODIFICA PRIMA, MASCHERA DOPO")
    print("="*60)
    
    # NOVITÀ: Decodifichiamo tutto subito in una lista di stringhe Latin-1
    latin1_packets = [hex_to_latin1(hp) for hp in hex_packets]
    
    input_texts = []
    target_texts = []
    
    for i in range(SEQUENCE_LENGTH):
        print(f"\n--- Elaborazione Posizione {i} ---")
        
        # Lavoriamo sulla lista già decodificata
        masked_packets = latin1_packets.copy()
        masked_packets[i] = "<extra_id_0>"
        
        # L'input è la stringa unita con lo spazio
        input_str = SEPARATOR.join(masked_packets)
        print(f"Input generato (Raw):  {repr(input_str)}")
        
        target_str = f"<extra_id_0> {latin1_packets[i]} <extra_id_1>"
        print(f"Target generato (Raw): {repr(target_str)}")
        
        input_texts.append(input_str)
        target_texts.append(target_str)

    print("\n" + "="*60)
    print(" FASE 2: TOKENIZZAZIONE (Cosa vede il modello ORA)")
    print("="*60)
    
    # Aumentiamo il max_length per vedere più token
    inputs = tokenizer(input_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    
    for i in range(SEQUENCE_LENGTH):
        print(f"\n--- Tensore di Input {i} ---")
        
        # Mostriamo i veri Token IDs (ignorando il pad)
        tokens = [t for t in inputs.input_ids[i].tolist() if t != 0]
        print(f"Token IDs: {tokens}")
        
        # Verifica cruciale: cerchiamo l'ID 258 (la maschera ByT5)
        if 259 in tokens:
            print(f"✅ Maschera <extra_id_0> (ID 258) TROVATA alla posizione {tokens.index(259)} del tensore!")
        else:
            print("❌ Maschera NON TROVATA!")

if __name__ == "__main__":
    main()