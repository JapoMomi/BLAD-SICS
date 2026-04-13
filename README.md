# 🛡️ BLAD-SICS: Byte-Level Anomaly Detection for Serial ICS

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)]()
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]() 

**BLAD-SICS** (Byte-Level Anomaly Detection for Serial ICS) is a novel, protocol-agnostic Intrusion Detection System (IDS) designed specifically for legacy serial communication channels (e.g., RS-232, RS-485, Modbus over Serial Line) in Industrial Control Systems (ICS). 

Instead of relying on Deep Packet Inspection (DPI) or handcrafted feature engineering, BLAD-SICS treats serial traffic as a continuous stream of raw bytes. By leveraging the token-free **ByT5 transformer architecture**, the system learns the structural grammar and temporal semantics of benign ICS communications to effectively detect zero-day threats.

---

## ✨ Key Features

* **🧩 Protocol-Agnostic Byte-Level Modeling:** Bypasses traditional subword tokenization by decoding raw hexadecimal payloads into Latin-1, allowing the ByT5 model to natively process and reconstruct raw network bytes.
* **⚖️ Dual-Model Ensemble Architecture:** 
   * *Single-Packet Syntax Detection:* Captures fine-grained structural anomalies (e.g., corrupted payloads, invalid checksums) using a sliding window masking approach.
   * *Sequence-Level Contextual Detection:* Captures sustained semantic attacks (e.g., complex response injections) using a leave-one-out masking strategy.
* **🛡️ Zero-Day Generalization:** Operates under a strictly one-class paradigm. The models are trained exclusively on benign traffic using Self-Supervised Learning (Masked Language Modeling) to establish a highly precise boundary of normality.
* **📉 High Noise Tolerance:** Integrates meta-classifiers (OCSVM, Isolation Forest, Random Forest) with Exponentially Weighted Moving Average (EWMA) to smooth transient physical noise and sensor jitter common in industrial serial buses.

---

## 🛠️ Technologies & Stack

* **Core Language:** Python
* **Deep Learning Framework:** PyTorch
* **Transformers:** Hugging Face `transformers` (ByT5-small)
* **Machine Learning & Thresholding:** Scikit-Learn
* **Data Engineering:** Pandas, NumPy
* **Hardware Acceleration:** CUDA/GPU support for intensive transformer training and inference

---

## 📊 Dataset

This framework was evaluated on an enhanced version of the **Gas Pipeline SCADA Dataset** (Morris et al.), featuring continuous Modbus over Serial Line traffic. The dataset contains over 214,000 Modbus frames and covers 35 distinct cyber-attack categories, including:
* Reconnaissance (Device/Function code scans)
* Response Injection (CMRI, NMRI)
* Command Injection (MSCI, MPCI, MFCI)
* Denial of Service (DoS)

---

## ⚙️ Prerequisites

To run this project, you will need:
* Python 3.8 or higher
* A CUDA-enabled GPU (Highly recommended for ByT5 training and inference)
* `pip` or `conda` package manager

---

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JapoMomi/BLAD-SICS.git
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```

3. [!IMPORTANT] Note on PyTorch Installation: 
Since BLAD-SICS relies on deep transformer architectures and is optimized for CUDA-enabled GPUs, we highly recommend installing the PyTorch version that specifically matches your hardware and CUDA drivers. Please follow the official PyTorch installation guide to get the correct command for your system before proceeding.

4. Install remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```
---

## 💻 Usage

The pipeline of this project is strictly divided into three main phases: Model Training, Score Reconstruction, and Anomaly Detection.

### Dataset Preparation
The raw dataset files are placed in the `Dataset/` directory. You can use the provided splitter scripts (e.g., `singlePacketSplitter.py` and `timeContextSplitter.py`) to generate the necessary data splits for the two respective architectures before proceeding.

### 1. Model Training
To train the underlying ByT5 language models exclusively on benign traffic, run the two training scripts. These processes will fine-tune the models for both syntactic and semantic analysis:

```bash
# Train the Single-Packet Syntax Model
python SingplePacketDetection/singlePacket-training.py

# Train the Time-Context Semantic Model
python TimeContextDetection/timeContext-training.py
```

### 2. Reconstruction Phase (Score Extraction)
Once the models are trained, you need to extract the reconstruction log-probabilities (anomaly scores). This phase evaluates the traffic and generates the `.csv` files required for the final classification.

**For Single-Packet & Dual-Model Detection:**
```bash
python3 DualModelDetection/dualModelReconstruction.py
```
*Note*: This generates dual_model_validation_results.csv and dual_model_detection_results.csv. These files are utilized by all single packet detection scripts.

**For Sequence-Level Detection:**
```bash
python3 TimeContextDetection/sequenceDetection/detection.py
```
*Note*: This generates detection_detailed_results_validation.csv and detection_detailed_results.csv inside the `sequenceDetection/` folder, which are explicitly used for evaluating entire sequences.

### 3. Anomaly Detection Phase
You can now apply the different classification strategies to the reconstructed scores. Navigate to the desired directory and run the specific detection script.

📝 ## Pro-Tip: Saving Outputs
To run a detection script and simultaneously save its output to the corresponding output/ directory (as structured in this repository), use the tee command. For example:
```bash
python3 -u DualModelDetection/oneClassClassifierDetection/OCSVM-ISODetection.py | tee DualModelDetection/oneClassClassifierDetection/output/ocsvm_results.txt
```

---
## ✒️ Author

**Jacopo Momesso**

**Institution:** University of Padova, Department of Mathematics "Tullio Levi-Civita"  

**Academic Year:** 2025-2026

**GitHub:** [@JapoMomi](https://github.com/JapoMomi)

### 👨‍🏫 Supervision
This research was conducted under the supervision of:
* **Supervisor:** Prof. Mauro Conti (University of Padova) 
* **Co-Supervisors:**  Prof. Fabio De Gaspari (Sapienza University of Rome), 
                       Dr. Denis Donadel (University of Verona) 

This project was developed as part of a Master Thesis in Cybersecurity.

---

## 📚 References & Acknowledgements

This project utilizes the enhanced Gas Pipeline SCADA dataset. If you use this dataset in your research, please ensure you cite the original authors:

* **Morris, T. H., Thornton, Z., & Turnipseed, I. P. (2015).** *Industrial control system simulation and data logging for intrusion detection system research.* In 7th International Conference on Critical Infrastructure Protection.

For researchers and developers, here is the BibTeX entry for the dataset reference:

```bibtex
@inproceedings{morris2015industrial,
  title={Industrial Control System Simulation and Data Logging for Intrusion Detection System Research},
  author={Thomas H. Morris and Zach Thornton and Ian P. Turnipseed},
  booktitle={7th International Conference on Critical Infrastructure Protection},
  year={2015},
  url={https://api.semanticscholar.org/CorpusID:42986835}
}
