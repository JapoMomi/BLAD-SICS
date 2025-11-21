import csv
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from config import N_PACKETS, MAX_LENGTH, MODEL_NAME


class DatasetBuilder:
    def __init__(self):
        self.n_packets = N_PACKETS
        #self.max_time_gap = MAX_TIME_GAP
        self.max_length = MAX_LENGTH
        self.model_name = MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def _load_packets(self, data_path):
        pkts, labels, ts = [], [], []
        with open(data_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                pkt = row[0].strip()
                try:
                    cat = int(row[1].strip())   
                    timestamp = float(row[-1])
                except ValueError:
                    continue
                pkts.append(pkt)
                labels.append(cat)
                ts.append(timestamp)
        return pkts, labels, ts

    def _group_sequences(self, packets, labels, timestamps):
        sequences, seq_labels = [], []
        start = 0
        while start < len(packets):
            seq = [packets[start]]
            seq_lbls = [labels[start]]
            #current_ts = timestamps[start]
            for j in range(start + 1, len(packets)):
                if len(seq) >= self.n_packets:
                    break
                #if timestamps[j] - current_ts > self.max_time_gap:
                #    break
                seq.append(packets[j])
                seq_lbls.append(labels[j])
                #current_ts = timestamps[j]
            seq_text = "     ".join(seq)
            seq_label = 1 if any(l != 0 for l in seq_lbls) else 0
            sequences.append(seq_text)
            seq_labels.append(seq_label)
            start += 1
        return sequences, seq_labels

    def _preprocess_function(self, examples):
        # 1. Tokenize Inputs and Targets (Clean)
        model_inputs = self.tokenizer(
            examples["input"], max_length=self.max_length, truncation=True, padding="max_length"
        )
        labels = self.tokenizer(
            examples["target"], max_length=self.max_length, truncation=True, padding="max_length"
        )
        
        input_ids = torch.tensor(model_inputs["input_ids"])
        
        # 2. Create Masking (Noise)
        # Probability of masking a token (e.g., 15% like BERT)
        mlm_probability = 0.15 
        
        # Create a random mask
        probability_matrix = torch.full(input_ids.shape, mlm_probability)
        
        # Don't mask special tokens (PAD, EOS, etc.)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
            for val in input_ids.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        # Create the boolean mask (True = Replace with <mask_token>)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 3. Apply Mask to Input
        # ByT5 sentinel/mask token is usually <extra_id_0>, <extra_id_1>... 
        # But for simple reconstruction, we can just use a specific noise token.
        # NOTE: ByT5 vocab is bytes. It doesn't have a [MASK] token like BERT.
        # Strategy: Replace with a random byte or a specific sentinel (e.g., ID 258 or 0).
        # A safe bet for ByT5 is using the padding token (0) or a specific '?' character if raw text.
        # Ideally, use tokenizer.mask_token_id if available, otherwise 0.
        mask_token_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else 0
        
        input_ids[masked_indices] = mask_token_id
        
        model_inputs["input_ids"] = input_ids.numpy().tolist()
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs

    def build_dataset(self, data_path):
        # Train dataset
        train_pkts, train_lbls, train_tmstmp = self._load_packets(f"{data_path}/train.txt")
        train_seq, train_seq_lbls = self._group_sequences(train_pkts, train_lbls, train_tmstmp)
        train_inputs = train_seq
        train_targets = train_seq
        train_dataset = Dataset.from_dict({"input": train_inputs, "target": train_targets})
        # Validation dataset
        val_pkts, val_lbls, val_tmstmp = self._load_packets(f"{data_path}/validation.txt")
        val_seq, val_seq_lbls = self._group_sequences(val_pkts, val_lbls, val_tmstmp)
        val_inputs = val_seq
        val_targets = val_seq
        val_dataset = Dataset.from_dict({"input": val_inputs, "target": val_targets})
        # Test dataset
        test_pkts, test_lbls, test_tmstmp = self._load_packets(f"{data_path}/validation.txt")
        test_seq, test_seq_lbls = self._group_sequences(test_pkts, test_lbls, test_tmstmp)
        test_inputs = test_seq
        test_targets = test_seq_lbls
        test_dataset = Dataset.from_dict({"input": test_inputs, "target": test_targets})

        dataset = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })

        tokenized_datasets = dataset.map(self._preprocess_function, batched=True, remove_columns=["input", "target"])
        return tokenized_datasets, dataset, self.tokenizer
