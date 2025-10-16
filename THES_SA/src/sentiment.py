import pandas as pd
import numpy as np
import logging
import nltk
from pathlib import Path
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from config import PRO, NEWS, TWEETS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK resources if not present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def analyze_news_sentiment(ticker):
    """
    Analyze sentiment of news articles for a ticker
    
    Args:
        ticker (str): Ticker symbol
    
    Returns:
        DataFrame: News data with sentiment scores
    """
    clean_ticker = ticker.replace('.', '_')
    
    # Path to processed news
    news_path = PRO / f"{clean_ticker}_news_processed.csv"
    
    if not news_path.exists():
        logger.error(f"Processed news file not found for {ticker}")
        return None
    
    logger.info(f"Analyzing sentiment for {ticker} news")
    
    # Load processed news
    news_df = pd.read_csv(news_path, parse_dates=['publishedAt'])
    
    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Apply sentiment analysis to title and description
    news_df['title_sentiment'] = news_df['title'].apply(
        lambda x: sid.polarity_scores(x)['compound'] if isinstance(x, str) else 0
    )
    
    if 'clean_description' in news_df.columns:
        news_df['desc_sentiment'] = news_df['clean_description'].apply(
            lambda x: sid.polarity_scores(x)['compound'] if isinstance(x, str) else 0
        )
    
    # Compute overall sentiment
    if 'clean_description' in news_df.columns:
        news_df['sentiment'] = (news_df['title_sentiment'] + news_df['desc_sentiment']) / 2
    else:
        news_df['sentiment'] = news_df['title_sentiment']
    
    # Aggregate by date
    daily_sentiment = news_df.groupby('date').agg({
        'sentiment': ['mean', 'std', 'count'],
        'title_sentiment': ['mean', 'std']
    })
    
    daily_sentiment.columns = ['sentiment_mean', 'sentiment_std', 'article_count', 
                              'title_sentiment_mean', 'title_sentiment_std']
    
    daily_sentiment.reset_index(inplace=True)
    
    # Save sentiment data
    output_path = PRO / f"{clean_ticker}_news_sentiment.csv"
    daily_sentiment.to_csv(output_path, index=False)
    logger.info(f"Saved news sentiment data to {output_path}")
    
    # Also save detailed sentiment data
    detailed_path = PRO / f"{clean_ticker}_news_detailed_sentiment.csv"
    news_df.to_csv(detailed_path, index=False)
    
    return daily_sentiment

def analyze_finbert_sentiment(ticker, use_gpu=False):
    """
    Analyze sentiment using FinBERT (more accurate for financial text)
    
    Args:
        ticker (str): Ticker symbol
        use_gpu (bool): Whether to use GPU for inference
    
    Returns:
        DataFrame: News data with FinBERT sentiment scores
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except ImportError:
        logger.error("Transformers or PyTorch not installed. Install with pip install transformers torch")
        return None
    
    clean_ticker = ticker.replace('.', '_')
    
    # Path to processed news
    news_path = PRO / f"{clean_ticker}_news_processed.csv"
    
    if not news_path.exists():
        logger.error(f"Processed news file not found for {ticker}")
        return None
    
    logger.info(f"Analyzing FinBERT sentiment for {ticker} news")
    
    # Load FinBERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load processed news
    news_df = pd.read_csv(news_path, parse_dates=['publishedAt'])
    
    # Define sentiment analysis function
    def get_finbert_sentiment(text):
        if not isinstance(text, str) or not text.strip():
            return 0
        
        # Truncate to prevent tokenizer overflow
        text = text[:512] if len(text) > 512 else text
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Get probabilities with softmax
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # FinBERT returns: negative (0), neutral (1), positive (2)
        # Convert to score between -1 and 1
        score = probs[0][2].item() - probs[0][0].item()  # positive - negative
        
        return score
    
    # Apply FinBERT to titles (less resource intensive than full content)
    logger.info("Processing titles with FinBERT...")
    news_df['finbert_title'] = news_df['title'].apply(get_finbert_sentiment)
    
    # Aggregate by date
    daily_sentiment = news_df.groupby('date').agg({
        'finbert_title': ['mean', 'std', 'count']
    })
    
    daily_sentiment.columns = ['finbert_mean', 'finbert_std', 'article_count']
    daily_sentiment.reset_index(inplace=True)
    
    # Save sentiment data
    output_path = PRO / f"{clean_ticker}_finbert_sentiment.csv"
    daily_sentiment.to_csv(output_path, index=False)
    logger.info(f"Saved FinBERT sentiment data to {output_path}")
    
    return daily_sentiment

def merge_sentiment_with_prices(ticker):
    """
    Merge sentiment data with stock price data
    
    Args:
        ticker (str): Ticker symbol
    
    Returns:
        DataFrame: Merged data with stock prices and sentiment
    """
    clean_ticker = ticker.replace('.', '_')
    
    # Path to processed stock data
    stock_path = PRO / f"{clean_ticker}_processed.csv"
    
    if not stock_path.exists():
        logger.error(f"Processed stock data not found for {ticker}")
        return None
    
    # Load stock data
    stock_df = pd.read_csv(stock_path, index_col=0, parse_dates=True)
    stock_df['date'] = stock_df.index.date
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    
    # Path to sentiment data
    sentiment_path = PRO / f"{clean_ticker}_news_sentiment.csv"
    
    if not sentiment_path.exists():
        logger.warning(f"News sentiment data not found for {ticker}")
        sentiment_df = None
    else:
        # Load sentiment data
        sentiment_df = pd.read_csv(sentiment_path, parse_dates=['date'])
    
    # Path to FinBERT sentiment data
    finbert_path = PRO / f"{clean_ticker}_finbert_sentiment.csv"
    
    if not finbert_path.exists():
        logger.warning(f"FinBERT sentiment data not found for {ticker}")
        finbert_df = None
    else:
        # Load FinBERT data
        finbert_df = pd.read_csv(finbert_path, parse_dates=['date'])
    
    # Merge stock data with sentiment data
    if sentiment_df is not None:
        merged_df = stock_df.merge(sentiment_df, on='date', how='left')
        merged_df[['sentiment_mean', 'sentiment_std', 'article_count']] = merged_df[
            ['sentiment_mean', 'sentiment_std', 'article_count']
        ].fillna(0)
    else:
        merged_df = stock_df.copy()
    
    # Merge with FinBERT data
    if finbert_df is not None:
        if sentiment_df is None:
            merged_df = stock_df.merge(finbert_df, on='date', how='left')
            merged_df[['finbert_mean', 'finbert_std', 'article_count']] = merged_df[
                ['finbert_mean', 'finbert_std', 'article_count']
            ].fillna(0)
        else:
            # Merge with finbert data, but don't duplicate article_count column
            finbert_df = finbert_df[['date', 'finbert_mean', 'finbert_std']]
            merged_df = merged_df.merge(finbert_df, on='date', how='left')
            merged_df[['finbert_mean', 'finbert_std']] = merged_df[
                ['finbert_mean', 'finbert_std']
            ].fillna(0)
    
    # Save merged data
    output_path = PRO / f"{clean_ticker}_prices_sentiment.csv"
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Saved merged price and sentiment data to {output_path}")
    
    return merged_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze sentiment in financial news")
    parser.add_argument("--ticker", required=True, help="Ticker symbol to analyze")
    parser.add_argument("--finbert", action="store_true", help="Use FinBERT for more accurate sentiment analysis")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for FinBERT inference")
    
    args = parser.parse_args()
    ticker = args.ticker
    
    # Analyze news sentiment with VADER
    analyze_news_sentiment(ticker)
    
    # Analyze with FinBERT if requested
    if args.finbert:
        analyze_finbert_sentiment(ticker, args.gpu)
    
    # Merge sentiment with price data
    merge_sentiment_with_prices(ticker)