import pandas as pd
import numpy as np
import json
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta
from config import RAW, PRO, NEWS, TWEETS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocess.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_stock_data(ticker):
    """
    Clean and preprocess stock data for a given ticker
    
    Args:
        ticker (str): Ticker symbol
    
    Returns:
        DataFrame: Cleaned stock data
    """
    # Replace dots in ticker for filename
    clean_ticker = ticker.replace('.', '_')
    filepath = RAW / f"{clean_ticker}_stock_data.csv"
    
    if not filepath.exists():
        logger.error(f"Stock data file not found: {filepath}")
        return None
    
    logger.info(f"Processing stock data for {ticker}")
    
    # Load data
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Handle missing values
    df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'], inplace=True)
    
    # Add technical indicators
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    df['ma_50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate daily change
    df['daily_change'] = df['Close'].pct_change()
    
    # Save processed data
    output_path = PRO / f"{clean_ticker}_processed.csv"
    df.to_csv(output_path)
    logger.info(f"Saved processed stock data to {output_path}")
    
    return df

def process_news_data(ticker):
    """
    Process news data for a given ticker
    
    Args:
        ticker (str): Ticker symbol
    
    Returns:
        DataFrame: Processed news data
    """
    # Replace dots in ticker for filename
    clean_ticker = ticker.replace('.', '_')
    
    # Find the most recent news file
    news_files = list(NEWS.glob(f"{clean_ticker}_news_*.json"))
    if not news_files:
        logger.error(f"No news files found for {ticker}")
        return None
    
    # Get most recent file
    latest_file = max(news_files, key=lambda x: x.stat().st_mtime)
    
    logger.info(f"Processing news data from {latest_file}")
    
    # Load news data
    with open(latest_file, 'r') as f:
        news_data = json.load(f)
    
    # Check if there are articles
    if 'articles' not in news_data or not news_data['articles']:
        logger.warning(f"No articles found in {latest_file}")
        return pd.DataFrame()
    
    # Create DataFrame from articles
    articles = news_data['articles']
    df = pd.DataFrame(articles)
    
    # Convert dates to datetime
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    
    # Extract date (without time)
    df['date'] = df['publishedAt'].dt.date
    
    # Clean text
    if 'content' in df.columns:
        df['clean_content'] = df['content'].apply(lambda x: clean_text(x) if isinstance(x, str) else '')
    
    if 'description' in df.columns:
        df['clean_description'] = df['description'].apply(lambda x: clean_text(x) if isinstance(x, str) else '')
    
    # Save processed data
    output_path = PRO / f"{clean_ticker}_news_processed.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed news data to {output_path}")
    
    return df

def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def merge_data_sources(ticker, stock_df=None, news_df=None, tweet_df=None):
    """
    Merge different data sources by date
    
    Args:
        ticker (str): Ticker symbol
        stock_df (DataFrame): Stock data
        news_df (DataFrame): News data
        tweet_df (DataFrame): Twitter data
    
    Returns:
        DataFrame: Merged data
    """
    clean_ticker = ticker.replace('.', '_')
    
    # Load stock data if not provided
    if stock_df is None:
        stock_path = PRO / f"{clean_ticker}_processed.csv"
        if not stock_path.exists():
            stock_df = clean_stock_data(ticker)
        else:
            stock_df = pd.read_csv(stock_path, index_col=0, parse_dates=True)
    
    if stock_df is None:
        logger.error(f"Stock data not available for {ticker}")
        return None
    
    # Create date index for merging
    stock_df['date'] = stock_df.index.date
    
    # Load news data if not provided
    if news_df is None:
        news_path = PRO / f"{clean_ticker}_news_processed.csv"
        if news_path.exists():
            news_df = pd.read_csv(news_path, parse_dates=['publishedAt'])
    
    # Aggregate news by date
    if news_df is not None and not news_df.empty and 'date' in news_df.columns:
        news_agg = news_df.groupby('date').size().reset_index(name='news_count')
        news_agg['date'] = pd.to_datetime(news_agg['date'])
        
        # Merge with stock data
        merged_df = stock_df.merge(news_agg, on='date', how='left')
        merged_df['news_count'] = merged_df['news_count'].fillna(0)
    else:
        merged_df = stock_df.copy()
        merged_df['news_count'] = 0
    
    # Process and merge tweet data if available
    if tweet_df is not None and not tweet_df.empty:
        # Process tweets
        pass
    
    # Save merged data
    output_path = PRO / f"{clean_ticker}_merged_data.csv"
    merged_df.to_csv(output_path)
    logger.info(f"Saved merged data to {output_path}")
    
    return merged_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess financial and sentiment data")
    parser.add_argument("--ticker", required=True, help="Ticker symbol to process")
    
    args = parser.parse_args()
    
    # Process all data for the given ticker
    ticker = args.ticker
    
    # Clean stock data
    stock_df = clean_stock_data(ticker)
    
    # Process news data
    news_df = process_news_data(ticker)
    
    # Merge data sources
    merged_df = merge_data_sources(ticker, stock_df, news_df)
    
    if merged_df is not None:
        logger.info(f"Successfully processed all data for {ticker}")
    else:
        logger.error(f"Failed to process data for {ticker}")