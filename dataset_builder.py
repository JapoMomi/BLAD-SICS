import csv
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from config import N_PACKETS, MAX_TIME_GAP, MAX_LENGTH, MODEL_NAME


class DatasetBuilder:
    def __init__(self, data_path):
        self.data_path = data_path
        self.n_packets = N_PACKETS
        self.max_time_gap = MAX_TIME_GAP
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
            current_ts = timestamps[start]
            for j in range(start + 1, len(packets)):
                if len(seq) >= self.n_packets:
                    break
                if timestamps[j] - current_ts > self.max_time_gap:
                    break
                seq.append(packets[j])
                seq_lbls.append(labels[j])
                current_ts = timestamps[j]
            seq_text = "     ".join(seq)
            seq_label = 1 if any(l != 0 for l in seq_lbls) else 0
            sequences.append(seq_text)
            seq_labels.append(seq_label)
            start += 1
        return sequences, seq_labels

    def _preprocess_function(self, examples):
        model_inputs = self.tokenizer(
            examples["input"], max_length=self.max_length, truncation=True
        )
        labels = self.tokenizer(
            examples["target"], max_length=self.max_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def build_dataset(self):
        # Train dataset
        train_pkts, train_lbls, train_tmstmp = self._load_packets(f"{self.data_path}/train.txt")
        train_seq, train_seq_lbls = self._group_sequences(train_pkts, train_lbls, train_tmstmp)
        train_inputs = train_seq
        train_targets = train_seq
        train_dataset = Dataset.from_dict({"input": train_inputs, "target": train_targets})
        # Validation dataset
        val_pkts, val_lbls, val_tmstmp = self._load_packets(f"{self.data_path}/validation.txt")
        val_seq, val_seq_lbls = self._group_sequences(val_pkts, val_lbls, val_tmstmp)
        val_inputs = val_seq
        val_targets = val_seq
        val_dataset = Dataset.from_dict({"input": val_inputs, "target": val_targets})
        
        dataset = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })

        tokenized_datasets = dataset.map(self._preprocess_function, batched=True, remove_columns=["input", "target"])
        return tokenized_datasets, dataset, self.tokenizer
