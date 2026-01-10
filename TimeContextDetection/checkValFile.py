import pandas as pd

# CONFIGURAZIONE
VAL_PATH = "/home/spritz/storage/disk0/Master_Thesis/Dataset_newVersion/splits/validation.txt"
SEPARATOR = ' '
SEQUENCE_LENGTH = 5

def check_data_integrity():
    print(f"Controllo integrità su: {VAL_PATH}")
    
    # Leggiamo il file
    try:
        df = pd.read_csv(VAL_PATH, header=None, names=['payload', 'label'], dtype=str)
    except Exception as e:
        print(f"ERRORE CRITICO lettura CSV: {e}")
        return

    bad_rows = 0
    total_rows = len(df)
    
    print(f"Totale righe trovate: {total_rows}")
    
    for idx, row in df.iterrows():
        payload = str(row['payload'])
        
        # Simuliamo la logica di split
        # Nota: La conversione hex->latin1 non cambia il numero di spazi, 
        # quindi possiamo contare gli spazi direttamene sulla stringa hex.
        packets = payload.strip().split(SEPARATOR)
        
        if len(packets) != SEQUENCE_LENGTH:
            bad_rows += 1
            if bad_rows <= 5: # Stampiamo solo i primi 5 errori per non intasare
                print(f"--- ERRORE RIGA {idx} ---")
                print(f"Pacchetti trovati: {len(packets)} (Attesi: {SEQUENCE_LENGTH})")
                print(f"Contenuto: '{payload}'")
    
    print("-" * 30)
    if bad_rows == 0:
        print("✅ TUTTO OK! Nessuna riga corrotta trovata.")
        print("Se escono ancora -100, il problema è nella tokenizzazione (pacchetti vuoti).")
    else:
        print(f"❌ TROVATE {bad_rows} RIGHE CORROTTE su {total_rows}!")
        print("Queste righe causano il -100 e sballano la threshold.")

if __name__ == "__main__":
    check_data_integrity()