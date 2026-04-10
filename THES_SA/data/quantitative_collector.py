"""
Quantitative Data Collector - Financial Time Series
Retrieves daily OHLCV data for Core Set and Benchmark Set using yfinance
(v2.0: Daily frequency only, intraday removed per advisor feedback)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import logging
from typing import List, Dict, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantitativeDataCollector:
    """Collects and stores daily financial time series data from Yahoo Finance"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize collector with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.core_set = self.config['tickers']['core_set']
        self.benchmark_set = self.config['tickers']['benchmark_set']
        self.all_tickers = self.config['tickers']['all_tickers']

        # Date configuration
        self.months_back = self.config['dates']['months_back']
        self.start_date = self.config['dates'].get('start_date')
        self.end_date = self.config['dates'].get('end_date')

        # Set up paths
        self.raw_path = Path(self.config['paths']['raw'])
        self.raw_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized collector for {len(self.all_tickers)} tickers")
        logger.info(f"Core Set: {self.core_set}")
        logger.info(f"Benchmark Set: {self.benchmark_set}")

    def _compute_dates(self) -> tuple:
        """Compute start and end dates based on configuration"""
        if self.start_date and self.end_date:
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
        else:
            end = datetime.now()
            start = end - timedelta(days=self.months_back * 30)

        logger.info(f"Date range: {start.date()} to {end.date()}")
        return start, end

    def fetch_daily_data(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for specified tickers

        Args:
            tickers: List of ticker symbols. If None, uses all_tickers from config

        Returns:
            DataFrame with multi-index (Date, Ticker) and OHLCV columns
        """
        if tickers is None:
            tickers = self.all_tickers

        start, end = self._compute_dates()
        logger.info(f"Fetching daily data for {len(tickers)} tickers...")

        all_data = []

        for ticker in tickers:
            try:
                logger.info(f"Downloading {ticker}...")
                stock = yf.Ticker(ticker)
                df = stock.history(
                    start=start,
                    end=end,
                    interval='1d',
                    auto_adjust=self.config['yfinance']['auto_adjust'],
                    prepost=self.config['yfinance']['prepost']
                )

                if df.empty:
                    logger.warning(f"No data returned for {ticker}")
                    continue

                df['Ticker'] = ticker
                df.reset_index(inplace=True)
                all_data.append(df)

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                continue

        if not all_data:
            logger.error("No data collected for any ticker!")
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])

        # Clean column names
        combined_df.columns = combined_df.columns.str.strip()

        logger.info(f"Collected {len(combined_df)} daily records")
        return combined_df

    def save_data(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV in raw data directory"""
        filepath = self.raw_path / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved data to {filepath}")

    def collect_all(self):
        """Collect daily data and save to disk"""
        logger.info("=" * 60)
        logger.info("Starting daily data collection")
        logger.info("=" * 60)

        # Collect daily data
        logger.info("\n--- Collecting Daily Data ---")
        daily_df = self.fetch_daily_data()
        if not daily_df.empty:
            self.save_data(daily_df, 'daily_ohlcv.csv')

        logger.info("\n" + "=" * 60)
        logger.info("Data collection complete!")
        logger.info("=" * 60)

    def get_ticker_info(self, ticker: str) -> Dict:
        """Get detailed information about a ticker"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'symbol': ticker,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'USD')
            }
        except Exception as e:
            logger.error(f"Error getting info for {ticker}: {e}")
            return {'symbol': ticker, 'error': str(e)}


def main():
    """Main execution function"""
    collector = QuantitativeDataCollector()

    # Get ticker information
    logger.info("\n--- Ticker Information ---")
    for ticker in collector.all_tickers:
        info = collector.get_ticker_info(ticker)
        logger.info(f"{ticker}: {info.get('name', 'N/A')} - {info.get('sector', 'N/A')}")

    # Collect daily data
    collector.collect_all()


if __name__ == "__main__":
    main()
