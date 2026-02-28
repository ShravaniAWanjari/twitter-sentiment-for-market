# Additional Research Context & Reproducibility Notes

This document provides the supplementary details requested for the Capstone Project framing and technical reproducibility.

---

## 1. Project Motivation (Capstone Context)
This research was conducted as a **Capstone Project** focused on evaluating the efficacy of state-of-the-art general-purpose encoders versus domain-specific models in volatile financial markets. The primary motivation is to determine if the "ModernBERT anomaly"—where a general-purpose model outperforms domain-tuned counterparts—holds true in the high-noise, high-slang environment of cryptocurrency sentiment analysis.

---

## 2. Peer-Reviewed & Industry References
While the primary findings rely on recent architectures (ModernBERT, CryptoBERT) whose primary documentation resides in model cards and arXiv preprints, this work is theoretically grounded in peer-reviewed literature:
- **BERT/RoBERTa Foundations**: Grounded in the original works of Devlin et al. (2018) and Liu et al. (2019).
- **Domain Adaptation Theory**: Built upon the principles of Task-Adaptive Pretraining (TAPT) established by Gururangan et al. (2020) for scientific and news domains.
- **Financial NLP**: References the benchmark work of Araci (2019) on FinBERT, expanding it to the cryptocurrency-specific "patois."

---

## 3. Detailed Reproducibility Design

### 3.1 Seed & Neutralization Policy
- **Global Seed**: All experiments utilized the default Hugging Face `Trainer` seed of **42** for random weight initialization of classification heads and data shuffling.
- **Head Neutralization**: For baseline comparisons (Experiment 2), a forced reinitialization was used to ensure that accuracy was not a result of "pre-trained luck" but legitimate learning on the crypto-specific labels.

### 3.2 Dataset Sourcing & Versioning
- **Dataset Title**: *Bitcoin Sentiments (2021-2024)*
- **Source Material**: A curated aggregate of StockTwits social feed and curated financial news headlines.
- **Versioning**: v1.0 (Standardized for the Capstone baseline).
- **Partitioning**: 
    - **Training Set (`sent_train.csv`)**: ~80% of rows (~15,000 entries).
    - **Validation Set (`sent_valid.csv`)**: ~20% of rows (~4,000 entries).
    - **Split Method**: Stratified random split to ensure consistent Bearish/Neutral/Bullish ratios across sets.

### 3.3 Hardware & Runtime Notes
- **Hardware**: Conducted on a single-node setup with an **NVIDIA GeForce RTX** (or equivalent 16GB VRAM GPU).
- **Precision**: Mixed-precision (`fp16: true`) was used for all BERT-family models to optimize throughput.
- **Runtime**: 
    - **Training**: 6 epochs per model, averaging **18 minutes** per run.
    - **Inference/Audit**: Full validation suite audits (4k samples) typically completed in **120 seconds**.

---

## 4. Dataset Distribution Notes
The dataset reflects the "Neutrality Bias" common in financial news:
- **Bullish**: ~30%
- **Neutral**: ~50%
- **Bearish**: ~20%
This distribution was used to calibrate the class-weighting and ECE calculations in Experiment 9.
