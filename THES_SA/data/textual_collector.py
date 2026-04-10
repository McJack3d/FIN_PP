"""
Textual Data Collector - News Headlines (v2.0)
Retrieves news headlines from FNSPID dataset (Hugging Face) and web scraping.
Updated keywords for v2.0 ticker universe.
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

    def _load_fnspid_local(self) -> pd.DataFrame:
        """
        Try to load FNSPID from a local path (Kaggle /kaggle/input/ or local CSV).
        The Kaggle FNSPID dataset has a single CSV at:
            modularization-demo/data/raw_analyst_ratings.csv
        with columns: headline, url, publisher, date, stock
        Returns empty DataFrame if not found.
        """
        kaggle_cfg = self.config.get('kaggle', {})
        fnspid_path = Path(kaggle_cfg.get('fnspid_input_path',
                                           '/kaggle/input/financial-news-and-stock-price-integration-dataset'))

        if not fnspid_path.exists():
            return pd.DataFrame()

        logger.info(f"Found local FNSPID at {fnspid_path}")

        csv_files = list(fnspid_path.glob('**/*.csv'))
        if not csv_files:
            logger.warning(f"No CSV files in {fnspid_path}")
            return pd.DataFrame()

        logger.info(f"  Found {len(csv_files)} CSV file(s): {[f.name for f in csv_files[:5]]}")

        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, on_bad_lines='skip')
                if len(df) > 0:
                    logger.info(f"  Loaded {len(df)} rows from {csv_file.name} — columns: {list(df.columns)}")
                    dfs.append(df)
            except Exception as e:
                logger.debug(f"Skipped {csv_file.name}: {e}")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined)} total articles from {len(dfs)} FNSPID file(s)")
        combined = self._standardize_columns(combined)

        # Parse dates — FNSPID has mixed formats (with and without timezone)
        if 'date' in combined.columns:
            combined['date'] = pd.to_datetime(combined['date'], format='mixed', utc=True, errors='coerce')
            combined['date'] = combined['date'].dt.tz_localize(None)
            valid_dates = combined['date'].notna().sum()
            logger.info(f"  Parsed {valid_dates}/{len(combined)} dates successfully")

        return combined

    def _filter_fnspid_for_tickers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter FNSPID articles relevant to our ticker universe.
        Uses both the 'symbol' column (exact match) and keyword search in 'text'.
        """
        ticker_keywords = {
            'SMR': ['SMR', 'NuScale', 'small modular reactor'],
            'LEU': ['LEU', 'Centrus Energy', 'uranium enrichment'],
            'LTBR': ['LTBR', 'Lightbridge'],
            'NXE': ['NXE', 'NexGen Energy'],
            'NNE': ['NNE', 'Nano Nuclear'],
            'LAC': ['LAC', 'Lithium Americas'],
            'CCJ': ['CCJ', 'Cameco'],
            'CEG': ['CEG', 'Constellation Energy'],
            'BWXT': ['BWXT', 'BWX Technologies'],
        }

        matched_dfs = []

        # Match by symbol column (exact ticker match)
        if 'symbol' in df.columns:
            for ticker in self.tickers:
                exact = df[df['symbol'] == ticker].copy()
                if not exact.empty:
                    exact['matched_ticker'] = ticker
                    matched_dfs.append(exact)
                    logger.info(f"  {ticker}: {len(exact)} articles by stock column")

        # Match by headline keywords
        if 'text' in df.columns:
            df['text'] = df['text'].astype(str)
            for ticker, keywords in ticker_keywords.items():
                pattern = '|'.join(keywords)
                keyword_matches = df[df['text'].str.contains(pattern, case=False, na=False, regex=True)].copy()
                if not keyword_matches.empty:
                    keyword_matches['matched_ticker'] = ticker
                    matched_dfs.append(keyword_matches)
                    logger.info(f"  {ticker}: {len(keyword_matches)} articles by keyword match")

        if not matched_dfs:
            logger.warning("No FNSPID articles matched any tickers")
            return pd.DataFrame()

        result = pd.concat(matched_dfs, ignore_index=True)
        if 'text' in result.columns:
            result = result.drop_duplicates(subset=['text'], keep='first')

        # Use matched_ticker as the symbol column
        if 'matched_ticker' in result.columns:
            result['symbol'] = result['matched_ticker']
            result.drop(columns=['matched_ticker'], inplace=True)

        logger.info(f"  Total matched: {len(result)} unique articles for {result['symbol'].nunique()} tickers")
        return result

    def fetch_fnspid_dataset(self) -> pd.DataFrame:
        """
        Fetch FNSPID (Financial News and Stock Price Integration Dataset).
        Checks local path first (Kaggle input), then falls back to Hugging Face.
        NOTE: FNSPID data is from 2009-2020. No date filtering is applied here —
        articles are filtered to relevant tickers and the merge phase handles
        date alignment with price data.

        Returns:
            DataFrame with news headlines and metadata
        """
        # Try local path first (Kaggle or manual download)
        local_df = self._load_fnspid_local()
        if not local_df.empty:
            # Filter to relevant tickers (by symbol + keyword matching)
            filtered = self._filter_fnspid_for_tickers(local_df)
            if not filtered.empty:
                logger.info(f"FNSPID: {len(filtered)} articles for our tickers "
                            f"(date range: {filtered['date'].min()} to {filtered['date'].max()})")
            return filtered

        logger.info("Fetching FNSPID dataset from Hugging Face...")

        try:
            from datasets import load_dataset

            # Load the dataset
            dataset = load_dataset(
                self.hf_dataset,
                cache_dir=str(self.hf_cache_dir),
            )

            # Convert to pandas DataFrame
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
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
        """Alternative methods to fetch FNSPID if main method fails"""
        # Try without trust_remote_code (already removed) and with different splits
        try:
            from datasets import load_dataset
            # Some FNSPID mirrors/forks exist - try common alternatives
            alternatives = [
                self.hf_dataset,
                "Zihan1004/FNSPID",
            ]
            for ds_name in alternatives:
                try:
                    logger.info(f"Trying dataset: {ds_name}...")
                    dataset = load_dataset(ds_name, cache_dir=str(self.hf_cache_dir))
                    split_name = list(dataset.keys())[0]
                    df = dataset[split_name].to_pandas()
                    logger.info(f"Loaded {len(df)} articles from {ds_name}")
                    df = self._standardize_columns(df)
                    return df
                except Exception:
                    continue
        except ImportError:
            pass

        # Try direct CSV download
        try:
            url = f"https://huggingface.co/datasets/{self.hf_dataset}/resolve/main/data.csv"
            logger.info(f"Attempting direct download from: {url}")
            df = pd.read_csv(url)
            logger.info(f"Loaded {len(df)} articles via direct download")
            df = self._standardize_columns(df)
            return df
        except Exception as e:
            logger.warning(f"Direct download failed: {e}")

        logger.warning("FNSPID dataset unavailable. Pipeline will rely on scraped news only.")
        return pd.DataFrame()

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different data sources"""
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
        Fetch news for a specific ticker using yfinance news API.
        Handles both old format (flat dict) and new format (nested 'content' dict).

        Args:
            ticker: Stock ticker symbol
            max_articles: Maximum number of articles to fetch

        Returns:
            List of dictionaries containing news data
        """
        logger.info(f"Fetching news for {ticker} via yfinance...")

        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            news = stock.news

            if not news:
                logger.info(f"No news found for {ticker}")
                return []

            # Log raw format of first item for debugging
            if news:
                sample = news[0]
                logger.info(f"  yfinance news format keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}")
                if isinstance(sample, dict) and 'content' in sample:
                    logger.info(f"  nested content keys: {list(sample['content'].keys())}")

            news_items = []
            for article in news[:max_articles]:
                try:
                    title, pub_date, source, url = self._parse_yf_article(article, ticker)
                    if title:
                        news_items.append({
                            'text': title,
                            'symbol': ticker,
                            'date': pub_date,
                            'source': source,
                            'url': url
                        })
                except Exception as e:
                    logger.debug(f"Error parsing article: {e}")
                    continue

            logger.info(f"Found {len(news_items)} articles with text for {ticker}")
            return news_items

        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []

    def _parse_yf_article(self, article: dict, ticker: str) -> tuple:
        """
        Parse a single yfinance news article, handling both old and new API formats.

        Old format (yfinance < 0.2.36):
            {'title': '...', 'link': '...', 'providerPublishTime': 1234567890, 'publisher': '...'}

        New format (yfinance >= 0.2.36):
            {'content': {'title': '...', 'pubDate': '...', 'provider': {'displayName': '...'},
                         'clickThroughUrl': {'url': '...'}, 'summary': '...'}, 'contentType': 'STORY'}

        Returns:
            (title, pub_date, source, url)
        """
        title = ''
        pub_date = datetime.now()
        source = 'yahoo_finance'
        url = ''

        # New format: nested under 'content'
        if 'content' in article and isinstance(article['content'], dict):
            content = article['content']
            title = content.get('title', '')
            source = content.get('provider', {}).get('displayName', 'yahoo_finance') if isinstance(content.get('provider'), dict) else content.get('provider', 'yahoo_finance')
            url = content.get('clickThroughUrl', {}).get('url', '') if isinstance(content.get('clickThroughUrl'), dict) else content.get('url', '')

            pub_str = content.get('pubDate', '')
            if pub_str:
                try:
                    pub_date = pd.to_datetime(pub_str, utc=True).to_pydatetime().replace(tzinfo=None)
                except Exception:
                    pub_date = datetime.now()

            # Use summary as fallback if title is empty
            if not title:
                title = content.get('summary', '')

        # Old format: flat dict
        else:
            title = article.get('title', '')
            source = article.get('publisher', 'yahoo_finance')
            url = article.get('link', '')

            publish_time = article.get('providerPublishTime')
            if publish_time and isinstance(publish_time, (int, float)):
                try:
                    pub_date = datetime.fromtimestamp(publish_time)
                except Exception:
                    pub_date = datetime.now()

        return title, pub_date, source, url

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

        # Keywords for nuclear energy and energy-transition stocks (v2.0)
        keywords = [
            "SMR nuclear", "small modular reactor", "NuScale Power",
            "Centrus Energy", "LEU nuclear",
            "Lightbridge nuclear", "LTBR",
            "NexGen Energy", "NXE uranium",
            "Nano Nuclear Energy", "NNE nuclear",
            "Lithium Americas", "LAC lithium",
            "Cameco uranium", "CCJ nuclear",
            "Constellation Energy", "CEG nuclear",
            "BWX Technologies", "BWXT nuclear",
            "nuclear energy stocks", "uranium mining"
        ]

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

    def scrape_google_news_rss(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch news headlines from Google News RSS feeds.
        No API key needed — public RSS endpoint.
        """
        if tickers is None:
            tickers = self.tickers

        logger.info(f"Fetching Google News RSS for {len(tickers)} tickers...")

        ticker_queries = {
            'SMR': 'NuScale+Power+SMR+stock',
            'LEU': 'Centrus+Energy+LEU+stock',
            'LTBR': 'Lightbridge+LTBR+stock',
            'NXE': 'NexGen+Energy+NXE+stock',
            'NNE': 'Nano+Nuclear+Energy+NNE+stock',
            'LAC': 'Lithium+Americas+LAC+stock',
            'CCJ': 'Cameco+CCJ+stock',
            'CEG': 'Constellation+Energy+CEG+stock',
            'BWXT': 'BWX+Technologies+BWXT+stock',
        }

        all_articles = []

        for ticker in tickers:
            query = ticker_queries.get(ticker, f'{ticker}+stock')
            rss_url = f'https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en'

            try:
                resp = requests.get(rss_url, timeout=15, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; research-bot)'
                })
                resp.raise_for_status()

                soup = BeautifulSoup(resp.content, 'xml')
                items = soup.find_all('item')

                for item in items:
                    title = item.find('title')
                    pub_date_tag = item.find('pubDate')
                    link = item.find('link')
                    source_tag = item.find('source')

                    title_text = title.get_text(strip=True) if title else ''
                    if not title_text:
                        continue

                    pub_date = datetime.now()
                    if pub_date_tag:
                        try:
                            pub_date = pd.to_datetime(pub_date_tag.get_text(), utc=True).to_pydatetime().replace(tzinfo=None)
                        except Exception:
                            pass

                    all_articles.append({
                        'text': title_text,
                        'symbol': ticker,
                        'date': pub_date,
                        'source': source_tag.get_text(strip=True) if source_tag else 'google_news',
                        'url': link.get_text(strip=True) if link else '',
                    })

                logger.info(f"  {ticker}: {len(items)} articles from Google News RSS")
                time.sleep(1)

            except Exception as e:
                logger.warning(f"  {ticker}: Google News RSS failed ({e})")
                continue

        if all_articles:
            df = pd.DataFrame(all_articles)
            logger.info(f"Google News RSS: {len(df)} total articles")
            return df

        return pd.DataFrame()

    def scrape_finviz_news(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scrape news headlines from Finviz ticker pages.
        """
        if tickers is None:
            tickers = self.tickers

        logger.info(f"Scraping Finviz news for {len(tickers)} tickers...")
        all_articles = []

        for ticker in tickers:
            url = f'https://finviz.com/quote.ashx?t={ticker}'
            try:
                resp = requests.get(url, timeout=15, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                resp.raise_for_status()

                soup = BeautifulSoup(resp.text, 'html.parser')
                news_table = soup.find(id='news-table')

                if not news_table:
                    logger.info(f"  {ticker}: no news table on Finviz")
                    continue

                rows = news_table.find_all('tr')
                current_date = datetime.now()

                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) < 2:
                        continue

                    date_cell = cols[0].get_text(strip=True)
                    link_tag = cols[1].find('a')

                    if not link_tag:
                        continue

                    title = link_tag.get_text(strip=True)
                    article_url = link_tag.get('href', '')

                    # Parse date — Finviz shows "Apr-07-26 09:30AM" or just "09:30AM"
                    try:
                        if len(date_cell) > 8:
                            current_date = pd.to_datetime(date_cell, errors='coerce')
                            if pd.isna(current_date):
                                current_date = datetime.now()
                    except Exception:
                        pass

                    all_articles.append({
                        'text': title,
                        'symbol': ticker,
                        'date': current_date,
                        'source': 'finviz',
                        'url': article_url,
                    })

                logger.info(f"  {ticker}: {len([a for a in all_articles if a['symbol'] == ticker])} articles from Finviz")
                time.sleep(2)  # Respect rate limits

            except Exception as e:
                logger.warning(f"  {ticker}: Finviz scraping failed ({e})")
                continue

        if all_articles:
            df = pd.DataFrame(all_articles)
            logger.info(f"Finviz: {len(df)} total articles")
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
        logger.info("Starting textual data collection (v2.0)")
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
            logger.info("\n--- Scraping Recent News (yfinance) ---")
            scraped_df = self.scrape_recent_news()
            if not scraped_df.empty:
                all_dataframes.append(scraped_df)
                self.save_data(scraped_df, 'scraped_news.csv')

            logger.info("\n--- Scraping Google News RSS ---")
            gnews_df = self.scrape_google_news_rss()
            if not gnews_df.empty:
                all_dataframes.append(gnews_df)
                self.save_data(gnews_df, 'google_news.csv')

            logger.info("\n--- Scraping Finviz News ---")
            finviz_df = self.scrape_finviz_news()
            if not finviz_df.empty:
                all_dataframes.append(finviz_df)
                self.save_data(finviz_df, 'finviz_news.csv')

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
