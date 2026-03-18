# THES_SA: Sentiment-Driven Nuclear Equity Forecasting (v2.0)

**Sentiment Analysis and Short-Term Return Predictability in Small-Cap Nuclear & Energy-Transition Equities: A FinBERT-LSTM Approach**

MSc Thesis - EDHEC Business School, Data Analytics & AI

---

## Quick Start

```bash
cd /path/to/THES_SA

# Automated (recommended)
./run_pipeline.sh

# Or run individual phases
python run_all.py --phase 0    # Feasibility audit only
python run_all.py --phase 1    # Data collection only
python run_all.py --phase 2    # Sentiment scoring only
python run_all.py --phase 3    # LSTM modeling only
python run_all.py --start-from 2  # Resume from Phase 2
```

---

## Project Structure

```
THES_SA/
├── config.yaml                    # Central configuration
├── requirements.txt               # Python dependencies
├── run_pipeline.sh               # Shell launcher
├── run_all.py                    # Master orchestrator
│
├── data/                         # Phase 0 + 1
│   ├── feasibility_audit.py     # Phase 0: FNSPID coverage audit
│   ├── pipeline.py              # Phase 1: Data collection orchestrator
│   ├── quantitative_collector.py # Daily OHLCV via yfinance
│   ├── textual_collector.py     # FNSPID + Yahoo Finance scraping
│   ├── preprocessing.py         # Feature engineering + normalization
│   ├── generate_summary.py      # Summary statistics & plots
│   ├── raw/                     # Raw collected data
│   ├── processed/               # Preprocessed data
│   └── news/                    # News articles
│
├── sentiment/                    # Phase 2
│   ├── scorer.py                # FinBERT sentiment scoring
│   └── features.py              # Daily Sentiment Index + Momentum
│
├── models/                       # Phase 3
│   ├── lstm_model.py            # Baseline + Augmented LSTM
│   ├── evaluation.py            # MAE, DA, Diebold-Mariano, H1/H2
│   └── explainability.py        # SHAP analysis
│
├── results/                      # All outputs
└── notebooks/                    # Exploration notebooks
```

---

## Ticker Universe

| Group | Tickers | Characteristics |
|-------|---------|-----------------|
| **Core Set (Small-Cap)** | SMR, LEU, LTBR, NXE, NNE, LAC | Market cap <$2B, high volatility, retail-driven |
| **Benchmark Set (Large-Cap)** | CCJ, CEG, BWXT | Market cap >$10B, institutional following |

---

## Pipeline Phases

### Phase 0: Feasibility Audit
Checks FNSPID news coverage per ticker. Tickers with <50 articles are flagged.

### Phase 1: Data Collection + Preprocessing
- **Quantitative**: Daily OHLCV via yfinance (12-month lookback)
- **Technical features**: RSI, MA(10,50), MACD, Realized Volatility
- **Target variables**: 1-day and 5-day forward log returns
- **Textual**: FNSPID dataset + Yahoo Finance scraping
- **Normalization**: Min-Max scaling per ticker

### Phase 2: Sentiment Analysis
- **FinBERT scoring**: Each headline scored (positive, negative, neutral)
- **Daily Sentiment Index**: Volume-weighted average per ticker per day
- **Sentiment Momentum**: 3-day rolling change in sentiment

### Phase 3: LSTM Modeling + Evaluation
- **Baseline LSTM**: Price/technical features only
- **Sentiment-Augmented LSTM**: Price + sentiment features
- **Evaluation**: MAE, Directional Accuracy, Diebold-Mariano test
- **Hypothesis Testing**:
  - H1: Sentiment improves prediction for small-cap nuclear stocks
  - H2: Improvement is larger for small-caps than large-caps (Sentiment Premium)
- **SHAP**: Feature importance analysis on augmented model

---

## Hypotheses

**H1**: An LSTM augmented with FinBERT sentiment achieves lower MAE and higher directional accuracy than a price-only baseline, for small-cap nuclear equities.

**H2 (Small-Cap Sentiment Premium)**: The predictive improvement from adding sentiment is significantly larger for small-cap nuclear stocks (Core Set) than for large-cap energy benchmarks (Benchmark Set).

---

## Requirements

- Python 3.8+
- TensorFlow >= 2.14 (LSTM models)
- PyTorch + Transformers (FinBERT)
- See `requirements.txt` for full list

---

## Author

**Alexandre Bredillot**
EDHEC Business School - MSc Data Analytics & AI

*Updated: March 2026 (v2.0)*
