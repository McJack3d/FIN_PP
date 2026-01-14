# 🧠 THES_SA: Sentiment-Driven Forecasting in Nuclear Energy Equities

**Sentiment-Driven Short-Term Forecasting in the Energy Equity Market**

This project implements a comprehensive data engineering and machine learning pipeline for financial forecasting, combining quantitative time-series analysis with sentiment analysis from financial news. The focus is on nuclear energy equities (SMR, LEU, NNE) compared against traditional energy benchmarks.

---

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
cd /Users/alexandrebredillot/Documents/GitHub/FIN_PP/THES_SA
./run_pipeline.sh
```

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data engineering pipeline
cd data
python pipeline.py --mode full
```

---

## 📁 Project Structure

```
THES_SA/
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
├── run_pipeline.sh              # Quick start script
├── README.md                    # This file
│
├── data/                        # Data engineering modules
│   ├── pipeline.py             # Main orchestrator
│   ├── quantitative_collector.py  # yfinance data collection
│   ├── textual_collector.py       # FNSPID + web scraping
│   ├── preprocessing.py           # Scaling & tokenization
│   ├── README.md               # Detailed documentation
│   │
│   ├── raw/                    # Raw collected data
│   ├── processed/              # Preprocessed data
│   ├── news/                   # News articles
│   └── hf_cache/              # Hugging Face cache
│
├── notebooks/                   # Jupyter notebooks
│   └── 01_data_engineering_pipeline.md
│
├── models/                      # Trained models (future)
└── results/                     # Analysis results (future)
```

---

## 🎯 Core Components

### Core Set (Nuclear Energy)
- **SMR** (NuScale Power) - Small Modular Reactor technology
- **LEU** (Centrus Energy) - Nuclear fuel enrichment
- **NNE** (Nano Nuclear Energy) - Next-generation nuclear energy

### Benchmark Set (Traditional Energy)
- XOM, CVX (Oil & Gas)
- NEE, DUK, SO, D (Utilities/Clean Energy)

---

## 🔧 Data Engineering Pipeline

### Phase 1: Data Collection

#### Quantitative Stream (yfinance)
- ✅ Daily OHLCV data for all tickers
- ✅ Intraday (1-hour) data for high-frequency analysis
- ✅ Historical data with configurable lookback period

#### Textual Stream
- ✅ **FNSPID Dataset**: Professional financial news from Hugging Face
- ✅ **Web Scraping**: Recent news from Yahoo Finance
- ✅ **NewsAPI** (optional): Additional news sources

### Phase 2: Data Preprocessing

#### Financial Data Processing
- Min-Max normalization (0-1 range for neural network stability)
- Technical indicators: MA, RSI, MACD, Bollinger Bands, Volatility
- Missing value handling and outlier detection

#### Textual Data Processing
- Text cleaning and normalization
- Tokenization for FinBERT
- **Stop-word preservation** (crucial for financial context)
- Deduplication and relevance filtering

---

## 📊 Usage Examples

```bash
# Run full pipeline
python data/pipeline.py --mode full

# Data collection only
python data/pipeline.py --mode collect --no-intraday

# Preprocessing only
python data/pipeline.py --mode preprocess

# With NewsAPI
python data/pipeline.py --mode full --use-newsapi
```

---

## 🧩 Methods & Tools

### Data Engineering
- **Quantitative**: yfinance, pandas, numpy
- **Textual**: Hugging Face datasets, BeautifulSoup, requests
- **Preprocessing**: scikit-learn (MinMaxScaler), transformers (FinBERT tokenizer)

### NLP & Sentiment Analysis
- **FinBERT**: Financial sentiment analysis
- **VADER, TextBlob**: Alternative sentiment methods
- **Transformers**: BERT-based models for context understanding

### ML Models (Planned)
- LSTM for time-series forecasting
- Random Forest, XGBoost for feature importance
- Transformer architectures for multi-modal learning

### Explainability (Planned)
- SHAP values for feature importance
- LIME for local interpretability
- Attention visualization for Transformers

---

## 📈 Output Data

### Processed Financial Data
- `daily_ohlcv_processed.csv`: Daily data with 20+ technical indicators
- `intraday_1h_ohlcv_processed.csv`: Hourly data
- `scalers.pkl`: Fitted Min-Max scalers for consistent normalization

### Processed Textual Data
- `news_processed.csv`: Cleaned and tokenized news (FinBERT-ready)
- `fnspid_news.csv`: FNSPID dataset articles
- `scraped_news.csv`: Recent scraped articles

---

## 📊 Expected Outcomes

- Quantitative insights on sentiment-price relationships in nuclear energy stocks
- Comparative analysis: news sentiment vs. price movements
- Transparent and interpretable forecasting framework
- Feature importance analysis across multiple data streams

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
tickers:
  core_set: [SMR, LEU, NNE]
  benchmark_set: [XOM, CVX, NEE, DUK, SO, D]

dates:
  months_back: 6

yfinance:
  intervals:
    daily: "1d"
    intraday: "1h"

huggingface:
  dataset: "oliverwang/FNSPID"
```

---

## 🔮 Roadmap

- [x] **Data Engineering Pipeline (Phase 1)**
  - [x] Quantitative data collection (yfinance)
  - [x] Textual data collection (FNSPID + scraping)
  - [x] Min-Max scaling and normalization
  - [x] FinBERT tokenization
- [ ] **Sentiment Analysis (Phase 2)**
  - [ ] FinBERT sentiment scoring
  - [ ] Sentiment-price alignment
  - [ ] Multi-source sentiment aggregation
- [ ] **Model Development (Phase 3)**
  - [ ] LSTM forecasting models
  - [ ] Multi-modal learning (price + sentiment)
  - [ ] Hyperparameter optimization
- [ ] **Explainability (Phase 4)**
  - [ ] SHAP values
  - [ ] Feature importance analysis
  - [ ] Attention visualization
- [ ] **Backtesting & Evaluation (Phase 5)**
  - [ ] Trading strategy simulation
  - [ ] Performance metrics
  - [ ] Risk-adjusted returns

---

## 🛠️ Requirements

- Python 3.8+
- yfinance
- transformers (Hugging Face)
- pandas, numpy, scikit-learn
- beautifulsoup4, requests
- PyTorch (for FinBERT)

See [requirements.txt](requirements.txt) for full list.

---

## 📚 Documentation

- [Data Engineering Details](data/README.md)
- [Example Notebook](notebooks/01_data_engineering_pipeline.md)
- [Configuration Guide](config.yaml)

---

## 🐛 Troubleshooting

### FNSPID Dataset Issues
```bash
pip install --upgrade datasets transformers
```

### Memory Constraints
```bash
python data/pipeline.py --no-intraday
```

### Rate Limiting
Built-in delays handle API rate limits automatically

---

## 👤 Author

**Alexandre Bredillot**  
EDHEC Business School  
Financial Engineering & Data Science

---

## 🙏 Acknowledgments

- **FNSPID Dataset**: oliverwang/FNSPID (Hugging Face)
- **FinBERT**: ProsusAI/finbert
- **yfinance**: Yahoo Finance API wrapper

---

*Last Updated: January 2026*
