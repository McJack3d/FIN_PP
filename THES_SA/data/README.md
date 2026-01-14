# Data Engineering Pipeline

This directory contains the complete data engineering pipeline for synchronizing quantitative and textual data streams for financial analysis.

## Overview

The pipeline collects and preprocesses two types of data:

1. **Quantitative Stream**: Daily and intraday OHLCV data from Yahoo Finance
2. **Textual Stream**: Financial news headlines from FNSPID dataset and web scraping

## Components

### Data Collection

- **`quantitative_collector.py`**: Fetches OHLCV data using yfinance for Core Set (SMR, LEU, NNE) and Benchmark Set
- **`textual_collector.py`**: Collects news headlines from FNSPID (Hugging Face) and web scraping

### Data Preprocessing

- **`preprocessing.py`**: 
  - Financial data: Min-Max scaling, technical indicators, feature engineering
  - Textual data: Tokenization for FinBERT (preserving stop-words for context)

### Pipeline Orchestration

- **`pipeline.py`**: Main orchestrator that runs the complete data engineering workflow

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/alexandrebredillot/Documents/GitHub/FIN_PP/THES_SA
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
cd data
python pipeline.py --mode full
```

### 3. Run Individual Phases

```bash
# Data collection only
python pipeline.py --mode collect

# Preprocessing only
python pipeline.py --mode preprocess
```

## Configuration

Edit `config.yaml` in the parent directory to customize:

- Ticker symbols (Core Set and Benchmark Set)
- Date ranges
- API keys (NewsAPI, if using)
- Data paths
- Processing parameters

## Output Structure

```
data/
├── raw/                          # Raw collected data
│   ├── daily_ohlcv.csv
│   ├── intraday_1h_ohlcv.csv
│   └── ...
├── processed/                    # Preprocessed data
│   ├── daily_ohlcv_processed.csv
│   ├── news_processed.csv
│   └── scalers.pkl              # Saved Min-Max scalers
├── news/                         # Raw news data
│   ├── fnspid_news.csv
│   ├── scraped_news.csv
│   └── all_news_combined.csv
└── hf_cache/                     # Hugging Face cache

```

## Command Line Options

```bash
python pipeline.py --help
```

Options:
- `--mode {full,collect,preprocess}`: Pipeline mode
- `--no-intraday`: Skip intraday data collection
- `--use-newsapi`: Enable NewsAPI (requires API key)
- `--no-scraping`: Disable web scraping
- `--config PATH`: Path to config file

## Individual Module Usage

### Quantitative Collector

```python
from quantitative_collector import QuantitativeDataCollector

collector = QuantitativeDataCollector()
collector.collect_all(include_intraday=True)
```

### Textual Collector

```python
from textual_collector import TextualDataCollector

collector = TextualDataCollector()
collector.collect_all(use_scraping=True)
```

### Preprocessing

```python
from preprocessing import FinancialDataPreprocessor, TextualDataPreprocessor

# Financial data
fin_proc = FinancialDataPreprocessor()
fin_proc.process_pipeline('daily_ohlcv.csv', 'daily_ohlcv_processed.csv')

# Textual data
text_proc = TextualDataPreprocessor()
text_proc.process_pipeline('all_news_combined.csv', 'news_processed.csv')
```

## Features

### Quantitative Processing
- ✅ Daily and intraday OHLCV data
- ✅ Min-Max normalization (0-1 range)
- ✅ Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- ✅ Volatility calculations
- ✅ Missing value handling
- ✅ Outlier detection and removal

### Textual Processing
- ✅ FNSPID dataset integration
- ✅ Web scraping for recent news
- ✅ Text cleaning and normalization
- ✅ FinBERT tokenization
- ✅ Stop-word preservation (important for financial context)
- ✅ Deduplication
- ✅ Relevance filtering

## Notes

- **FNSPID Dataset**: Automatically cached locally after first download
- **Rate Limiting**: Built-in delays to respect API rate limits
- **Error Handling**: Robust error handling with detailed logging
- **Scalability**: Parallel processing support for multiple tickers
- **Reproducibility**: Saved scalers ensure consistent normalization

## Troubleshooting

### FNSPID Dataset Not Loading

If the Hugging Face dataset fails to load:
```bash
pip install --upgrade datasets transformers
```

### Memory Issues with Intraday Data

Intraday data can be large. Skip with:
```bash
python pipeline.py --no-intraday
```

### NewsAPI Rate Limits

Free tier has limits. Use web scraping as primary source:
```bash
python pipeline.py --no-newsapi
```

## Next Steps

After running the pipeline, processed data is ready for:
1. Sentiment analysis with FinBERT
2. Feature engineering and alignment
3. Model training (LSTM, Transformer, etc.)
4. Backtesting and evaluation
