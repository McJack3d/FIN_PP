"""
Textual Data Collector - News Headlines
Retrieves news headlines from FNSPID dataset (Hugging Face) and web scraping
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import logging
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextualDataCollector:
    """Collects news headlines from FNSPID and web sources"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize collector with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tickers = self.config['tickers']['all_tickers']
        self.core_set = self.config['tickers']['core_set']
        
        # Date configuration
        self.months_back = self.config['dates']['months_back']
        self.start_date = self.config['dates'].get('start_date')
        self.end_date = self.config['dates'].get('end_date')
        
        # Hugging Face configuration
        self.hf_dataset = self.config['huggingface']['dataset']
        self.hf_cache_dir = Path(self.config['huggingface']['cache_dir'])
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up paths
        self.news_path = Path(self.config['paths']['news'])
        self.news_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized textual collector for {len(self.tickers)} tickers")
    
    def _compute_dates(self) -> tuple:
        """Compute start and end dates based on configuration"""
        if self.start_date and self.end_date:
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
        else:
            end = datetime.now()
            start = end - timedelta(days=self.months_back * 30)
        
        return start, end
    
    def fetch_fnspid_dataset(self) -> pd.DataFrame:
        """
        Fetch FNSPID (Financial News and Stock Price Integration Dataset) from Hugging Face
        
        Returns:
            DataFrame with news headlines and metadata
        """
        logger.info("Fetching FNSPID dataset from Hugging Face...")
        
        try:
            from datasets import load_dataset
            
            # Load the dataset
            dataset = load_dataset(
                self.hf_dataset,
                cache_dir=str(self.hf_cache_dir),
                trust_remote_code=True
            )
            
            # Convert to pandas DataFrame
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                # If no train split, use the first available split
                split_name = list(dataset.keys())[0]
                df = dataset[split_name].to_pandas()
            
            logger.info(f"Loaded {len(df)} articles from FNSPID dataset")
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Filter by date range
            start, end = self._compute_dates()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df[(df['date'] >= start) & (df['date'] <= end)]
                logger.info(f"Filtered to {len(df)} articles in date range")
            
            return df
            
        except ImportError:
            logger.error("datasets library not installed. Install with: pip install datasets")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading FNSPID dataset: {e}")
            logger.info("Attempting alternative approach...")
            return self._fetch_fnspid_alternative()
    
    def _fetch_fnspid_alternative(self) -> pd.DataFrame:
        """Alternative method to fetch FNSPID if main method fails"""
        try:
            # Try to load from Hugging Face datasets hub directly
            url = f"https://huggingface.co/datasets/{self.hf_dataset}/resolve/main/data.csv"
            logger.info(f"Attempting to download from: {url}")
            
            df = pd.read_csv(url)
            logger.info(f"Loaded {len(df)} articles via direct download")
            
            df = self._standardize_columns(df)
            return df
            
        except Exception as e:
            logger.error(f"Alternative fetch also failed: {e}")
            return pd.DataFrame()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different data sources"""
        # Common column name mappings
        column_mappings = {
            'headline': 'text',
            'headlines': 'text',
            'title': 'text',
            'news': 'text',
            'article': 'text',
            'published': 'date',
            'publish_date': 'date',
            'timestamp': 'date',
            'ticker': 'symbol',
            'stock': 'symbol',
            'company': 'symbol'
        }
        
        # Rename columns (case insensitive)
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in column_mappings:
                df.rename(columns={col: column_mappings[col_lower]}, inplace=True)
        
        return df
    
    def scrape_recent_news(self, tickers: Optional[List[str]] = None, 
                           max_articles: int = 100) -> pd.DataFrame:
        """
        Scrape recent news for tickers from financial news websites
        
        Args:
            tickers: List of ticker symbols
            max_articles: Maximum articles to scrape per ticker
            
        Returns:
            DataFrame with scraped news headlines
        """
        if tickers is None:
            tickers = self.tickers
        
        logger.info(f"Scraping recent news for {len(tickers)} tickers...")
        
        all_news = []
        
        for ticker in tickers:
            try:
                news_items = self._scrape_ticker_news(ticker, max_articles)
                all_news.extend(news_items)
                time.sleep(2)  # Rate limiting
            except Exception as e:
                logger.error(f"Error scraping news for {ticker}: {e}")
                continue
        
        if not all_news:
            logger.warning("No news articles scraped")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_news)
        logger.info(f"Scraped {len(df)} total news articles")
        return df
    
    def _scrape_ticker_news(self, ticker: str, max_articles: int) -> List[Dict]:
        """
        Scrape news for a specific ticker from Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol
            max_articles: Maximum number of articles to scrape
            
        Returns:
            List of dictionaries containing news data
        """
        logger.info(f"Scraping news for {ticker}...")
        
        try:
            # Using Yahoo Finance RSS feed
            url = f"https://finance.yahoo.com/quote/{ticker}/news"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            news_items = []
            
            # Find news articles (structure may vary)
            articles = soup.find_all('h3', class_='Mb(5px)')[:max_articles]
            
            for article in articles:
                try:
                    headline = article.get_text().strip()
                    
                    # Try to find link and timestamp
                    link_elem = article.find_parent('a')
                    link = link_elem.get('href') if link_elem else None
                    
                    news_items.append({
                        'text': headline,
                        'symbol': ticker,
                        'date': datetime.now(),
                        'source': 'yahoo_finance',
                        'url': f"https://finance.yahoo.com{link}" if link else None
                    })
                except Exception as e:
                    logger.debug(f"Error parsing article: {e}")
                    continue
            
            logger.info(f"Found {len(news_items)} articles for {ticker}")
            return news_items
            
        except Exception as e:
            logger.error(f"Error scraping Yahoo Finance for {ticker}: {e}")
            return []
    
    def fetch_newsapi_data(self, api_key: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch news from NewsAPI (if configured)
        
        Args:
            api_key: NewsAPI key (optional, will use config if not provided)
            
        Returns:
            DataFrame with news articles
        """
        if api_key is None:
            api_key = self.config.get('newsapi', {}).get('api_key')
        
        if not api_key or api_key == "":
            logger.warning("No NewsAPI key configured, skipping NewsAPI fetch")
            return pd.DataFrame()
        
        logger.info("Fetching news from NewsAPI...")
        
        start, end = self._compute_dates()
        all_articles = []
        
        # Keywords for nuclear energy stocks
        keywords = ["SMR nuclear", "small modular reactor", "Centrus Energy", 
                   "LEU nuclear", "Nano Nuclear", "nuclear energy"]
        
        base_url = self.config['newsapi']['url']
        
        for keyword in keywords:
            try:
                params = {
                    'q': keyword,
                    'from': start.strftime('%Y-%m-%d'),
                    'to': end.strftime('%Y-%m-%d'),
                    'language': 'en',
                    'sortBy': 'relevancy',
                    'apiKey': api_key,
                    'pageSize': 100
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('status') == 'ok':
                    articles = data.get('articles', [])
                    for article in articles:
                        all_articles.append({
                            'text': article.get('title', ''),
                            'description': article.get('description', ''),
                            'date': pd.to_datetime(article.get('publishedAt')),
                            'source': article.get('source', {}).get('name', 'newsapi'),
                            'url': article.get('url'),
                            'keyword': keyword
                        })
                    logger.info(f"Found {len(articles)} articles for '{keyword}'")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching NewsAPI data for '{keyword}': {e}")
                continue
        
        if all_articles:
            df = pd.DataFrame(all_articles)
            df = df.drop_duplicates(subset=['text'])
            logger.info(f"Collected {len(df)} unique articles from NewsAPI")
            return df
        
        return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV in news data directory"""
        filepath = self.news_path / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved news data to {filepath}")
    
    def collect_all(self, use_newsapi: bool = False, use_scraping: bool = True):
        """
        Collect all textual data and save to disk
        
        Args:
            use_newsapi: Whether to fetch from NewsAPI
            use_scraping: Whether to use web scraping
        """
        logger.info("=" * 60)
        logger.info("Starting textual data collection")
        logger.info("=" * 60)
        
        all_dataframes = []
        
        # Fetch FNSPID dataset
        logger.info("\n--- Fetching FNSPID Dataset ---")
        fnspid_df = self.fetch_fnspid_dataset()
        if not fnspid_df.empty:
            fnspid_df['source'] = 'fnspid'
            all_dataframes.append(fnspid_df)
            self.save_data(fnspid_df, 'fnspid_news.csv')
        
        # Fetch from NewsAPI if enabled
        if use_newsapi:
            logger.info("\n--- Fetching from NewsAPI ---")
            newsapi_df = self.fetch_newsapi_data()
            if not newsapi_df.empty:
                all_dataframes.append(newsapi_df)
                self.save_data(newsapi_df, 'newsapi_news.csv')
        
        # Scrape recent news if enabled
        if use_scraping:
            logger.info("\n--- Scraping Recent News ---")
            scraped_df = self.scrape_recent_news()
            if not scraped_df.empty:
                all_dataframes.append(scraped_df)
                self.save_data(scraped_df, 'scraped_news.csv')
        
        # Combine all sources
        if all_dataframes:
            logger.info("\n--- Combining All News Sources ---")
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Remove duplicates based on text similarity
            if 'text' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
            
            combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
            combined_df = combined_df.sort_values('date', ascending=False)
            
            self.save_data(combined_df, 'all_news_combined.csv')
            logger.info(f"Combined dataset: {len(combined_df)} unique articles")
        
        logger.info("\n" + "=" * 60)
        logger.info("Textual data collection complete!")
        logger.info("=" * 60)


def main():
    """Main execution function"""
    collector = TextualDataCollector()
    collector.collect_all(use_newsapi=False, use_scraping=True)


if __name__ == "__main__":
    main()
