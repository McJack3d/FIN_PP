import requests
import pandas as pd
import json
import time
import os
import logging
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
from config import NEWSAPI_KEY, NEWSAPI_URL, GDELT_URL, TICKERS, START, END, NEWS, TWEETS, RAW, TWEET_QUERY_TEMPLATE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_news(query, from_date=None, to_date=None, max_retries=3, sleep_time=1):
    """
    Fetch news articles using NewsAPI
    Note: Free plan only allows access to articles from the last 30 days
    
    Args:
        query (str): Search query
        from_date (str): Start date in YYYY-MM-DD format (defaults to 30 days ago)
        to_date (str): End date in YYYY-MM-DD format (defaults to today)
        max_retries (int): Maximum number of retry attempts
        sleep_time (int): Seconds to wait between retries
        
    Returns:
        dict: JSON response from NewsAPI
    """
    # Default to last 30 days if no dates provided (free plan limitation)
    if from_date is None:
        from_date = (datetime.now() - timedelta(days=29)).strftime('%Y-%m-%d')
    if to_date is None:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    # Ensure dates are within the allowed range (last 30 days)
    thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if from_date < thirty_days_ago:
        logger.warning(f"Adjusting from_date from {from_date} to {thirty_days_ago} (NewsAPI free plan limitation)")
        from_date = thirty_days_ago
        
    logger.info(f"Fetching news for query '{query}' from {from_date} to {to_date}")
    
    params = {
        'q': query,
        'from': from_date,
        'to': to_date,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 100,  # Maximum page size
        'apiKey': NEWSAPI_KEY
    }
    
    # Implement retries with exponential backoff
    for attempt in range(max_retries):
        try:
            response = requests.get(NEWSAPI_URL, params=params, timeout=10)
            response.raise_for_status()  # Raise exception for 4XX/5XX status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = sleep_time * (2 ** attempt)  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error("Max retries exceeded. Returning empty response.")
                return {"status": "error", "message": str(e)}
    
    return {"status": "error", "message": "Unknown error occurred"}

def fetch_gdelt(query, max_retries=3, sleep_time=1, timeout=30):  # Increased timeout
    """
    Fetch news articles from GDELT Global Knowledge Graph
    
    Args:
        query (str): Search query
        max_retries (int): Maximum number of retry attempts
        sleep_time (int): Seconds to wait between retries
        
    Returns:
        dict: JSON response from GDELT
    """
    logger.info(f"Fetching GDELT data for query '{query}'")
    
    params = {
        'query': query, 
        'mode': 'ArtList', 
        'format': 'JSON',
        'maxrecords': 100  # Reduced from 250 to make requests faster
    }
    
    # Implement retries with exponential backoff
    for attempt in range(max_retries):
        try:
            logger.info(f"GDELT attempt {attempt+1}/{max_retries} with timeout={timeout}s")
            response = requests.get(GDELT_URL, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching GDELT data (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = sleep_time * (2 ** attempt) + 1  # Increased wait time
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error("Max retries exceeded. Returning empty response.")
                # Return empty articles to avoid further errors
                return {"articles": []}
    
    return {"articles": []}  # Return empty articles list instead of error message

def fetch_gdelt_chunked(query, start_date, end_date, chunk_days=5):
    """
    Fetch GDELT data in smaller date chunks to avoid timeouts
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    logger.info(f"Fetching chunked GDELT data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Initialize results with empty articles list
    results = {"articles": []}
    
    # Process in chunks of days
    current_date = start_date
    while current_date <= end_date:
        # Calculate end of current chunk
        next_date = min(current_date + timedelta(days=chunk_days), end_date)
        
        # Create date-specific query
        date_query = f"{query} sourcePublishDate:{current_date.strftime('%Y%m%d')}:{next_date.strftime('%Y%m%d')}"
        
        # Fetch data for this chunk
        chunk_results = fetch_gdelt(date_query)
        
        # Add articles from this chunk to results
        if "articles" in chunk_results and chunk_results["articles"]:
            results["articles"].extend(chunk_results["articles"])
            logger.info(f"Added {len(chunk_results['articles'])} articles from chunk {current_date.strftime('%Y-%m-%d')}")
        
        # Move to next chunk
        current_date = next_date + timedelta(days=1)
        
        # Sleep between chunks to avoid rate limiting
        if current_date <= end_date:
            time.sleep(2)
    
    return results

def fetch_tweets(keywords, start=None, end=None, limit=500):
    """
    Fetch tweets using snscrape
    
    Args:
        keywords (str or list): Keywords to search for
        start (str): Start date in YYYY-MM-DD format
        end (str): End date in YYYY-MM-DD format
        limit (int): Maximum number of tweets to fetch
        
    Returns:
        DataFrame: Pandas DataFrame containing tweets
    """
    if start is None:
        start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
        
    logger.info(f"Fetching tweets for keywords '{keywords}' from {start} to {end}")
    
    try:
        import snscrape.modules.twitter as sntwitter
        
        # Create search query
        if isinstance(keywords, list):
            keywords_str = " OR ".join([f'"{k}"' for k in keywords])
        else:
            keywords_str = f'"{keywords}"'
            
        query = TWEET_QUERY_TEMPLATE.format(
            keywords=keywords_str,
            start=start,
            end=end
        )
        
        logger.info(f"Using Twitter query: {query}")
        
        # Scrape tweets with progress bar
        tweets = []
        for i, tweet in tqdm(enumerate(sntwitter.TwitterSearchScraper(query).get_items()), 
                             desc="Fetching tweets", total=limit):
            if i >= limit:
                break
            tweets.append({
                "date": tweet.date,
                "user": tweet.user.username,
                "content": tweet.rawContent,
                "retweets": tweet.retweetCount,
                "likes": tweet.likeCount,
                "urls": [url for url in tweet.outlinks] if hasattr(tweet, 'outlinks') else [],
                "hashtags": [tag.lower() for tag in tweet.hashtags] if hasattr(tweet, 'hashtags') else []
            })
            
        # Create DataFrame
        df = pd.DataFrame(tweets)
        logger.info(f"Fetched {len(df)} tweets")
        return df
        
    except ImportError:
        logger.error("snscrape not available. Please install with: pip install snscrape")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching tweets: {str(e)}")
        return pd.DataFrame()

def fetch_stock_data(tickers=TICKERS, start=START, end=END):
    """
    Fetch stock data using yfinance
    
    Args:
        tickers (list): List of ticker symbols
        start (str): Start date
        end (str): End date
        
    Returns:
        dict: Dictionary of DataFrames with ticker as key
    """
    logger.info(f"Fetching stock data for {len(tickers)} tickers from {start} to {end}")
    
    try:
        import yfinance as yf
        
        stock_data = {}
        for ticker in tqdm(tickers, desc="Fetching stocks"):
            try:
                # Download data
                data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
                
                if data.empty:
                    logger.warning(f"No data found for {ticker}")
                    continue
                
                # Add ticker column
                data['ticker'] = ticker
                
                # Calculate returns - handle different column names
                if 'Adj Close' in data.columns:
                    price_col = 'Adj Close'
                else:
                    price_col = 'Close'
                    logger.info(f"Using 'Close' instead of 'Adj Close' for {ticker}")
                
                # Calculate returns using available price column
                data['return'] = data[price_col].pct_change()
                data['cumulative_return'] = (1 + data['return']).cumprod() - 1
                
                # Calculate volatility (20-day rolling std)
                data['volatility'] = data['return'].rolling(window=20).std()
                
                # Save to dictionary
                stock_data[ticker] = data
                
                # Save to CSV
                clean_ticker = ticker.replace('.', '_')
                csv_path = RAW / f"{clean_ticker}_stock_data.csv"
                data.to_csv(csv_path)
                logger.info(f"Saved {ticker} data to {csv_path}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {str(e)}")
        
        return stock_data
        
    except ImportError:
        logger.error("yfinance not available. Please install with: pip install yfinance")
        return {}

def save_data(data, filename, directory=None, format='json'):
    """
    Save data to file
    
    Args:
        data: Data to save
        filename (str): Filename
        directory (Path): Directory to save to (defaults to RAW)
        format (str): File format (json or csv)
    """
    if directory is None:
        directory = RAW
        
    # Create directory if it doesn't exist
    directory.mkdir(parents=True, exist_ok=True)
    
    # Create full path
    file_path = directory / filename
    
    try:
        if format.lower() == 'json':
            with open(file_path, 'w') as f:
                json.dump(data, f)
        elif format.lower() == 'csv':
            data.to_csv(file_path)
        else:
            logger.error(f"Unsupported format: {format}")
            return
            
        logger.info(f"Saved data to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")

def fetch_all_data(tickers=TICKERS, fetch_stocks=True, fetch_news_data=True, fetch_gdelt_data=True, fetch_twitter_data=True):
    """
    Fetch all data for all tickers
    
    Args:
        tickers (list): List of ticker symbols
        fetch_stocks (bool): Whether to fetch stock data
        fetch_news_data (bool): Whether to fetch news data
        fetch_gdelt_data (bool): Whether to fetch GDELT data
        fetch_twitter_data (bool): Whether to fetch Twitter data
    """
    logger.info(f"Starting comprehensive data fetch for {len(tickers)} tickers")
    
    # Fetch stock data
    if fetch_stocks:
        logger.info("Fetching stock data...")
        stock_data = fetch_stock_data(tickers)
        logger.info(f"Fetched stock data for {len(stock_data)} tickers")
    
    # Get date ranges that respect API limitations
    today = datetime.now().strftime('%Y-%m-%d')
    one_month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Process each ticker
    for ticker in tickers:
        logger.info(f"Processing ticker: {ticker}")
        
        # Company name for broader searches (remove exchange suffix if present)
        company_name = ticker.split('.')[0]
        
        # 1. Fetch NewsAPI data
        if fetch_news_data:
            logger.info(f"Fetching NewsAPI data for {ticker}...")
            news_data = fetch_news(query=f"{ticker} OR {company_name}", 
                                   from_date=one_month_ago, 
                                   to_date=today)
                
            if 'articles' in news_data:
                article_count = len(news_data['articles'])
                logger.info(f"Found {article_count} news articles for {ticker}")
                
                # Save data
                filename = f"{ticker.replace('.', '_')}_news_{datetime.now().strftime('%Y%m%d')}.json"
                save_data(news_data, filename, directory=NEWS)
            else:
                logger.warning(f"No news articles found for {ticker}: {news_data}")
        
        # 2. Fetch GDELT data
        if fetch_gdelt_data:
            logger.info(f"Fetching GDELT data for {ticker}...")
            gdelt_data = fetch_gdelt(query=f"{ticker} OR {company_name}")
            
            if 'articles' in gdelt_data:
                article_count = len(gdelt_data['articles'])
                logger.info(f"Found {article_count} GDELT articles for {ticker}")
                
                # Save data
                filename = f"{ticker.replace('.', '_')}_gdelt_{datetime.now().strftime('%Y%m%d')}.json"
                save_data(gdelt_data, filename, directory=NEWS)
            else:
                logger.warning(f"No GDELT articles found for {ticker}")
        
        # 3. Fetch Twitter data
        if fetch_twitter_data:
            logger.info(f"Fetching Twitter data for {ticker}...")
            tweet_data = fetch_tweets(keywords=[ticker, company_name], 
                                      start=one_month_ago, 
                                      end=today, 
                                      limit=1000)
            
            if not tweet_data.empty:
                logger.info(f"Found {len(tweet_data)} tweets for {ticker}")
                
                # Save data
                filename = f"{ticker.replace('.', '_')}_tweets_{datetime.now().strftime('%Y%m%d')}.csv"
                save_data(tweet_data, filename, directory=TWEETS, format='csv')
            else:
                logger.warning(f"No tweets found for {ticker}")
        
        logger.info(f"Completed processing for {ticker}")
        
    logger.info("Data fetch complete!")

if __name__ == "__main__":
    # Create command-line interface
    parser = argparse.ArgumentParser(description="Fetch financial and sentiment data for analysis")
    parser.add_argument("--tickers", nargs="+", default=TICKERS, help="List of ticker symbols")
    parser.add_argument("--no-stocks", action="store_true", help="Skip fetching stock data")
    parser.add_argument("--no-news", action="store_true", help="Skip fetching news data")
    parser.add_argument("--no-gdelt", action="store_true", help="Skip fetching GDELT data")
    parser.add_argument("--no-twitter", action="store_true", help="Skip fetching Twitter data")
    parser.add_argument("--single", help="Fetch data for a single ticker only")
    
    args = parser.parse_args()
    
    if args.single:
        fetch_all_data(
            tickers=[args.single], 
            fetch_stocks=not args.no_stocks, 
            fetch_news_data=not args.no_news,
            fetch_gdelt_data=not args.no_gdelt,
            fetch_twitter_data=not args.no_twitter
        )
    else:
        fetch_all_data(
            tickers=args.tickers,
            fetch_stocks=not args.no_stocks,
            fetch_news_data=not args.no_news,
            fetch_gdelt_data=not args.no_gdelt,
            fetch_twitter_data=not args.no_twitter
        )