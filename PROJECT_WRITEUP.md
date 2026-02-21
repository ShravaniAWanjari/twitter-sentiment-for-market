# Capstone Progress Writeup

## ML Work Completed
- Built a shared model wrapper (`CryptoTransformerBase`) for consistent tokenization, training, and XAI‑ready access to embeddings/attention.
- Trained and evaluated five models on the bitcoin sentiment dataset (ModernBERT, CryptoBERT, FinBERT, BERT‑base, RoBERTa‑base).
- Added robust preprocessing and label alignment for a 3‑class schema (Bearish/Neutral/Bullish).
- Implemented error analysis tooling:
  - Misclassification extraction with confidence and top‑k probabilities.
  - Error rates by text length, slang/URL/handle signals, and confidence bins.
  - Confusion‑pair breakdowns with per‑pair CSVs.
- Built a generic trainer (`experiments/train_model.py`) to train any registered model consistently.
- Expanded the model registry to six models:
  - ModernBERT, CryptoBERT, FinBERT, DeBERTa‑v3 (small), BERT‑base, RoBERTa‑base.
- Built an interactive Streamlit dashboard (`streamlit_app.py`) that consolidates benchmarks, error analysis, and saved plots.

## Quant Finance Work Completed
- Focused on domain‑specific sentiment classification for crypto market data (bitcoin news/social text).
- Standardized a 3‑class sentiment mapping aligned to market interpretation:
  - Bearish (negative), Neutral, Bullish (positive).
- Evaluated model performance with finance‑relevant signals:
  - Slang robustness, latency per tweet, and confusion patterns that matter for trading sentiment.
- Laid groundwork to compare models for deployment trade‑offs:
  - Accuracy vs. latency and macro vs. weighted scores.
- Prepared outputs suitable for dashboarding and monitoring drift via error analysis CSVs.

## Deliverables Produced
- Trained model artifacts for ModernBERT (`experiments/modernbert_runs/final_model`).
- Benchmark and evaluation CSVs for visualization and reporting.
- Streamlit dashboard (`streamlit_app.py`) with interactive charts, tables, and saved plot images.
- Error analysis outputs in `results/` for drill‑downs and dashboards.

## Challenges and Mitigations
- Model variability (e.g., DeBERTa using `token_type_ids` and unstable gradients) broke a fully plug‑and‑play pipeline.
- Mitigations added:
  - Base `forward` now accepts optional `token_type_ids` and extra kwargs for model‑specific inputs.
  - Training keeps all dataset columns (`remove_unused_columns=False`) so labels and extra fields are preserved.
  - Per‑model training profiles set safer defaults (LR, max length, fp16).
  - Optional preflight check auto‑disables fp16 and clamps LR if gradients are non‑finite.
  - Swapped DeBERTa‑v3 base to the smaller variant to improve stability on the dataset.
- Remaining challenge: schema differences across models (e.g., DeBERTa input signatures and task heads) still require per‑model handling to guarantee 100% plug‑and‑play.

## Current Benchmark Snapshot (from `benchmark_results.csv`)
- ModernBERT: macro F1 ~0.363, accuracy ~0.366.
- CryptoBERT: macro F1 ~0.351, accuracy ~0.425.
- FinBERT: macro F1 ~0.143, accuracy ~0.150.
- BERT‑base: macro F1 ~0.288, accuracy ~0.377.
- RoBERTa‑base: macro F1 ~0.188, accuracy ~0.394.

## Next Steps
- Re‑run benchmarks after any hyperparameter tuning to update `benchmark_results.csv`.
- Generate additional confusion matrices for CryptoBERT and RoBERTa if needed.
- Stress‑test on out‑of‑sample windows to validate robustness across market regimes.
