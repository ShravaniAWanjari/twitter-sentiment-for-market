# Quant NLP Research Project: Technical Overview

This document provides a comprehensive breakdown of the **Quant NLP Research Platform**, covering methodology, technical implementation, and architectural details.

---

##  1. Core Functionalities

### 1.1 Model Training & Ranking (Leaderboard)
- **Multi-Model Pipeline**: Trains and benchmarks five different transformer architectures: `ModernBERT`, `CryptoBERT`, `FinBERT`, `BERT-Base`, and `RoBERTa-Base`.
- **Dynamic Leaderboard**: Ranks models based on **F1 Macro Score**, Accuracy, and Latency.
- **Auto-Selection**: The backtest engine automatically detects and promotes the "Best Performer" based on the highest F1 Macro score.

### 1.2 Backtest Simulation Engine
- **Strategy Gating**: Unlike standard backtests, this platform uses **NLP Gating** (discussed in detail below) to filter trade signals.
- **Risk Management**: Implements per-trade risk scaling (e.g., 2% of current balance) and initial capital simulation.
- **Performance Metrics**: Calculates Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, and Profit Factor for both baseline (unfiltered) and gated (NLP-filtered) strategies.
- **Visualization**: Equity curves and drawdown profiles are synchronized to show exactly where the NLP layer saved or generated capital.

### 1.3 Deep Explainability
- **Triple-Method Attribution**: For any headline, the system runs three simultaneous explainability tests:
    1. **Occlusion**: Masking tokens to measure marginal impact.
    2. **Integrated Gradients (IG)**: Path-based gradient aggregation for causal credit.
    3. **Gradient × Input**: High-speed directional sensitivity analysis.
- **Stability Analysis**: Measures how robust a prediction is when the most important words are removed.
- **Counterfactual Testing**: Automatically attempts to "flip" the model's prediction by swapping key drivers with their logic antonyms.

### 1.4 Historical Case Studies
- **Contextual Analysis**: Fetches real historical Bitcoin headlines from the backtest period (via NewsData.io) and runs them through the selected model to show real-world gating outcomes.

---

##  2. Methodology

### 2.1 Avoiding Lookahead Bias
To ensure the backtest represents real-world execution, we enforce strict temporal constraints:
- **T-1 Signal Origin**: The trade signal (Technical + NLP) is generated at the close of Day $T-1$.
- **T Execution**: Market returns are only applied on Day $T$.
- **Code implementation**: `sig = signal_series.iloc[i-1]; price_change = returns[i]`. The signal from the previous period (past) determines exposure to the next period (future).

### 2.2 Lag Period
- **Daily Frequency**: The model operates on daily bars. 
- **1-Day Decision Lag**: The cumulative sentiment for a day is used to gate the *next* day’s technical signal. This accounts for the time required to aggregate, process, and execute on news data.

### 2.3 NLP Gating: Beyond Direct Sentiment
A critical design choice in this project is **not using raw sentiment labels** (Bullish/Bearish) as the primary filter. 
- **The Logic**: A "Bullish" headline doesn't always lead to a price surge if the model's confidence is low.
- **What we use instead**: **Confidence-Weighted Gating**.
- **The Filter**: The technical signal (e.g., RSI Buy) is only allowed to enter the market if the model's **Prediction Confidence (Certainty)** exceeds a user-defined threshold.
- **Purpose**: This effectively acts as a "Certainty Filter." It prevents the strategy from executing on technical signals during periods of high news volatility or model ambiguity.

---

## 🛠️ 3. Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Backend** | Python 3.10+, FastAPI |
| **Frontend** | React, Vite, Vanila CSS |
| **AI/NLP** | PyTorch, HuggingFace Transformers, ModernBERT |
| **Data Processing**| Pandas, NumPy, Scikit-learn |
| **Finance API** | yfinance (Market data), NewsData.io (Real-time news) |
| **Generative AI** | Google Gemini (Grounded research chat) |
| **Explainability** | Custom implementation (Occlusion, IG, Grad×Input) |

---

## 📡 4. API Reference

### Configuration & Status
- `GET /api/config`: Returns available models and dataset paths.
- `GET /api/status`: Polls the current state of training/benchmarking (JobManager stats).
- `GET /api/models/trained`: Checks which models have weights, benchmarks, and analysis ready.
- `POST /api/clear-session`: Resets research state (optional weight deletion).

### Benchmarking & Training
- `POST /api/train`: Starts the five-model pipeline (Background task).
- `GET /api/benchmark`: Returns the precision/recall/f1 metrics for all models.
- `GET /api/errors/summaries/all`: Pulls detailed error attributions (length, signal, confidence).

### Backtesting
- `POST /api/backtest`: Runs a full walk-forward simulation with NLP gating.
- `GET /api/backtest/latest`: Retrieves the last simulation artifact.
- `POST /api/backtest/headline-samples`: Fetches and analyzes historical news for the backtest date range.

### NLP & Content
- `POST /api/analyze`: Bulk sentiment analysis for a list of strings.
- `POST /api/news/analyze-headline`: Single-headline analysis with full triples-explainability.
- `GET /api/news/bitcoin-headlines`: Live top-5 Bitcoin headlines.
- `POST /api/chat`: Grounded RAG-style chat using backtest results and Gemini.

---

## 📦 5. Project Directory Structure
```text
/backend          # API cores & simulation engines
/dataset          # CSVs (Historical prices & sentiment labels)
/experiments      # Model weights & checkpoint data
/frontend         # React source + design tokens
/results          # Generated CSVs, images, and run artifacts
```

---
*This analysis is for research purposes only.*
