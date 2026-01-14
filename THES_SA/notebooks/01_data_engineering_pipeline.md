# Data Engineering Pipeline - Example Usage

This notebook demonstrates how to use the data engineering pipeline components.

## Setup

```python
import sys
from pathlib import Path

# Add data directory to path
sys.path.append(str(Path.cwd().parent / 'data'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from quantitative_collector import QuantitativeDataCollector
from textual_collector import TextualDataCollector
from preprocessing import FinancialDataPreprocessor, TextualDataPreprocessor
from pipeline import DataEngineeringPipeline

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
sns.set_style('darkgrid')
```

## 1. Quick Start - Run Full Pipeline

```python
# Initialize pipeline
pipeline = DataEngineeringPipeline(config_path='../config.yaml')

# Run complete pipeline
pipeline.run_full_pipeline(
    include_intraday=True,
    use_newsapi=False,
    use_scraping=True
)
```

## 2. Individual Component Usage

### 2.1 Quantitative Data Collection

```python
# Initialize collector
quant_collector = QuantitativeDataCollector(config_path='../config.yaml')

# Get ticker information
for ticker in quant_collector.core_set:
    info = quant_collector.get_ticker_info(ticker)
    print(f"{ticker}: {info.get('name')} - {info.get('sector')}")

# Fetch daily data
daily_data = quant_collector.fetch_daily_data()
print(f"\nCollected {len(daily_data)} daily records")
print(daily_data.head())

# Fetch intraday data
intraday_data = quant_collector.fetch_intraday_data(interval='1h', period='1mo')
print(f"\nCollected {len(intraday_data)} intraday records")
```

### 2.2 Textual Data Collection

```python
# Initialize collector
text_collector = TextualDataCollector(config_path='../config.yaml')

# Fetch FNSPID dataset
fnspid_data = text_collector.fetch_fnspid_dataset()
print(f"FNSPID dataset: {len(fnspid_data)} articles")
print(fnspid_data.head())

# Scrape recent news
recent_news = text_collector.scrape_recent_news(max_articles=50)
print(f"\nScraped {len(recent_news)} recent articles")
```

### 2.3 Financial Data Preprocessing

```python
# Initialize preprocessor
fin_preprocessor = FinancialDataPreprocessor(config_path='../config.yaml')

# Load and clean data
raw_data = fin_preprocessor.load_raw_data('daily_ohlcv.csv')
clean_data = fin_preprocessor.clean_financial_data(raw_data)

# Add technical features
data_with_features = fin_preprocessor.add_technical_features(clean_data)
print(f"\nAdded features: {list(data_with_features.columns)}")

# Normalize data
normalized_data = fin_preprocessor.normalize_data(data_with_features)
print(f"\nNormalized {len(normalized_data)} records")

# Save processed data
fin_preprocessor.save_processed_data(normalized_data, 'example_processed.csv')
fin_preprocessor.save_scalers('example_scalers.pkl')
```

### 2.4 Textual Data Preprocessing

```python
# Initialize preprocessor
text_preprocessor = TextualDataPreprocessor(config_path='../config.yaml')

# Load news data
news_data = text_preprocessor.load_news_data('all_news_combined.csv')

# Deduplicate
news_dedup = text_preprocessor.deduplicate_news(news_data)

# Filter relevant articles
news_filtered = text_preprocessor.filter_relevant_news(news_dedup)

# Tokenize for FinBERT
news_tokenized = text_preprocessor.tokenize_for_finbert(news_filtered)
print(f"\nTokenized {len(news_tokenized)} articles")
print(f"Average tokens: {news_tokenized['token_count'].mean():.1f}")
```

## 3. Data Analysis and Visualization

### 3.1 Analyze Price Data

```python
# Load processed data
processed_data = pd.read_csv('../data/processed/daily_ohlcv_processed.csv')
processed_data['Date'] = pd.to_datetime(processed_data['Date'])

# Plot price trends for Core Set
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

for idx, ticker in enumerate(['SMR', 'LEU', 'NNE']):
    ticker_data = processed_data[processed_data['Ticker'] == ticker].sort_values('Date')
    
    axes[idx].plot(ticker_data['Date'], ticker_data['Close'], label='Close Price')
    axes[idx].plot(ticker_data['Date'], ticker_data['MA_20'], label='MA 20', alpha=0.7)
    axes[idx].set_title(f'{ticker} - Price Trend')
    axes[idx].set_ylabel('Price')
    axes[idx].legend()
    axes[idx].grid(True)

plt.tight_layout()
plt.show()
```

### 3.2 Volatility Analysis

```python
# Calculate and plot volatility
fig, ax = plt.subplots(figsize=(15, 6))

for ticker in ['SMR', 'LEU', 'NNE']:
    ticker_data = processed_data[processed_data['Ticker'] == ticker].sort_values('Date')
    ax.plot(ticker_data['Date'], ticker_data['Volatility_20d'], label=ticker)

ax.set_title('20-Day Volatility Comparison (Core Set)')
ax.set_xlabel('Date')
ax.set_ylabel('Volatility')
ax.legend()
ax.grid(True)
plt.show()
```

### 3.3 News Coverage Analysis

```python
# Load processed news
news_data = pd.read_csv('../data/processed/news_processed.csv')
news_data['date'] = pd.to_datetime(news_data['date'])

# News volume over time
news_by_date = news_data.groupby(news_data['date'].dt.date).size()

plt.figure(figsize=(15, 6))
plt.plot(news_by_date.index, news_by_date.values)
plt.title('News Article Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
```

### 3.4 Correlation Analysis

```python
# Correlation between tickers
pivot_data = processed_data.pivot(index='Date', columns='Ticker', values='Daily_Return')
correlation = pivot_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1)
plt.title('Return Correlation Matrix')
plt.tight_layout()
plt.show()
```

## 4. Data Quality Checks

```python
def check_data_quality(df, name):
    """Check data quality metrics"""
    print(f"\n{'='*60}")
    print(f"Data Quality Report: {name}")
    print(f"{'='*60}")
    
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}" if 'Date' in df.columns else "No date column")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        print("\nMissing values:")
        print(missing[missing > 0])
    else:
        print("\n✓ No missing values")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    # Numerical statistics
    if 'Close' in df.columns:
        print(f"\nPrice statistics:")
        print(df.groupby('Ticker')['Close'].describe())

# Check financial data
financial_data = pd.read_csv('../data/processed/daily_ohlcv_processed.csv')
financial_data['Date'] = pd.to_datetime(financial_data['Date'])
check_data_quality(financial_data, "Financial Data")

# Check news data
news_data = pd.read_csv('../data/processed/news_processed.csv')
print(f"\n{'='*60}")
print(f"News Data Quality Report")
print(f"{'='*60}")
print(f"Total articles: {len(news_data)}")
print(f"Average text length: {news_data['cleaned_text'].str.len().mean():.1f} characters")
print(f"Sources: {news_data['source'].value_counts().to_dict()}")
```

## 5. Export for Model Training

```python
def prepare_for_training(financial_df, news_df, save_path='../data/processed/training_ready.csv'):
    """Prepare synchronized dataset for model training"""
    
    # Ensure date columns are datetime
    financial_df['Date'] = pd.to_datetime(financial_df['Date'])
    news_df['date'] = pd.to_datetime(news_df['date'])
    
    # Aggregate news by date
    news_aggregated = news_df.groupby('date').agg({
        'cleaned_text': lambda x: ' '.join(x),
        'text': 'count'
    }).rename(columns={'text': 'news_count'})
    
    # Merge financial and news data
    merged_df = financial_df.merge(
        news_aggregated,
        left_on='Date',
        right_index=True,
        how='left'
    )
    
    # Fill missing news counts
    merged_df['news_count'] = merged_df['news_count'].fillna(0)
    
    # Save
    merged_df.to_csv(save_path, index=False)
    print(f"✓ Saved training-ready dataset to {save_path}")
    print(f"  - {len(merged_df)} records")
    print(f"  - {len(merged_df.columns)} features")
    
    return merged_df

# Prepare training data
training_data = prepare_for_training(financial_data, news_data)
print(training_data.head())
```

## 6. Custom Pipeline Configuration

```python
# Run custom pipeline with specific parameters
custom_pipeline = DataEngineeringPipeline(config_path='../config.yaml')

# Only collect Core Set data
custom_pipeline.quant_collector.all_tickers = custom_pipeline.quant_collector.core_set

# Run with custom settings
custom_pipeline.run_data_collection(
    collect_quantitative=True,
    collect_textual=True,
    include_intraday=False,  # Skip intraday for faster execution
    use_newsapi=False,
    use_scraping=True
)
```

## Summary

This notebook demonstrates:
- ✅ Full pipeline execution
- ✅ Individual component usage
- ✅ Data visualization and analysis
- ✅ Data quality checks
- ✅ Export for model training

Next steps:
1. Implement sentiment analysis with FinBERT
2. Build predictive models (LSTM, Transformer)
3. Backtest trading strategies
4. Evaluate model performance
