"""
Sentiment Feature Engineering (Phase 2b)
Computes Daily Sentiment Index and Sentiment Momentum per ticker.
Merges sentiment features with quantitative data into a single dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentFeatureEngineer:
    """Aggregates sentiment scores into daily features and merges with price data"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.processed_path = Path(self.config['paths']['processed'])
        self.momentum_window = self.config['sentiment']['momentum_window']
        self.all_tickers = self.config['tickers']['all_tickers']

        logger.info("SentimentFeatureEngineer initialized")

    def compute_daily_sentiment_index(self, df_scored: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a Daily Sentiment Index per ticker.
        Uses volume-weighted average of sentiment_score if article count > 1,
        otherwise simple mean.

        Args:
            df_scored: DataFrame with sentiment scores and date/ticker info

        Returns:
            DataFrame with columns: Date, Ticker, Sentiment_Index, Article_Count
        """
        logger.info("Computing Daily Sentiment Index...")

        df = df_scored.copy()

        # Ensure date column
        date_col = 'date' if 'date' in df.columns else 'Date'
        if date_col not in df.columns:
            logger.error(f"No date column found in scored data. Columns: {list(df.columns)}")
            return pd.DataFrame()

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df['date_only'] = df[date_col].dt.date

        # Identify ticker column
        ticker_col = None
        for candidate in ['symbol', 'Ticker', 'ticker']:
            if candidate in df.columns:
                ticker_col = candidate
                break

        daily_records = []

        if ticker_col and df[ticker_col].notna().any():
            # Group by date and ticker
            for (date, ticker), group in df.groupby(['date_only', ticker_col]):
                if ticker not in self.all_tickers:
                    continue
                daily_records.append({
                    'Date': pd.to_datetime(date),
                    'Ticker': ticker,
                    'Sentiment_Index': group['sentiment_score'].mean(),
                    'Sentiment_Std': group['sentiment_score'].std() if len(group) > 1 else 0.0,
                    'Article_Count': len(group),
                    'Positive_Ratio': (group['sentiment_score'] > 0.05).mean(),
                    'Negative_Ratio': (group['sentiment_score'] < -0.05).mean(),
                })
        else:
            # No ticker column - assign articles to tickers via keyword matching
            logger.info("No ticker column found. Matching articles to tickers via keywords.")
            daily_records = self._match_and_aggregate(df)

        if not daily_records:
            logger.warning("No daily sentiment records produced")
            return pd.DataFrame()

        df_daily = pd.DataFrame(daily_records)
        logger.info(f"Computed daily sentiment for {len(df_daily)} date-ticker pairs")
        return df_daily

    def _match_and_aggregate(self, df: pd.DataFrame) -> list:
        """Match articles to tickers by keyword and aggregate per day"""
        text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'

        ticker_keywords = {
            'SMR': ['SMR', 'NuScale'],
            'LEU': ['LEU', 'Centrus'],
            'LTBR': ['LTBR', 'Lightbridge'],
            'NXE': ['NXE', 'NexGen'],
            'NNE': ['NNE', 'Nano Nuclear'],
            'LAC': ['LAC', 'Lithium Americas'],
            'CCJ': ['CCJ', 'Cameco'],
            'CEG': ['CEG', 'Constellation Energy'],
            'BWXT': ['BWXT', 'BWX Technologies'],
        }

        records = []
        for ticker, keywords in ticker_keywords.items():
            pattern = '|'.join(keywords)
            matches = df[df[text_col].str.contains(pattern, case=False, na=False, regex=True)]

            if matches.empty:
                continue

            for date, group in matches.groupby('date_only'):
                records.append({
                    'Date': pd.to_datetime(date),
                    'Ticker': ticker,
                    'Sentiment_Index': group['sentiment_score'].mean(),
                    'Sentiment_Std': group['sentiment_score'].std() if len(group) > 1 else 0.0,
                    'Article_Count': len(group),
                    'Positive_Ratio': (group['sentiment_score'] > 0.05).mean(),
                    'Negative_Ratio': (group['sentiment_score'] < -0.05).mean(),
                })

        return records

    def add_sentiment_momentum(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add Sentiment Momentum: rolling change in Sentiment_Index over N days.

        Args:
            df_daily: DataFrame with daily sentiment index

        Returns:
            DataFrame with Sentiment_Momentum column added
        """
        logger.info(f"Computing Sentiment Momentum (window={self.momentum_window})...")

        df = df_daily.copy()
        df = df.sort_values(['Ticker', 'Date'])

        for ticker in df['Ticker'].unique():
            mask = df['Ticker'] == ticker
            ticker_data = df.loc[mask, 'Sentiment_Index']
            df.loc[mask, 'Sentiment_Momentum'] = ticker_data.diff(self.momentum_window)

        df['Sentiment_Momentum'] = df['Sentiment_Momentum'].fillna(0.0)
        return df

    def merge_with_price_data(self,
                               df_sentiment: pd.DataFrame,
                               price_file: str = 'daily_ohlcv_processed.csv') -> pd.DataFrame:
        """
        Merge daily sentiment features with processed price data.
        Left-joins on (Date, Ticker) so all price rows are kept.
        Days without news get Sentiment_Index=0, Article_Count=0.

        Args:
            df_sentiment: Daily sentiment DataFrame
            price_file: Processed price CSV filename

        Returns:
            Merged DataFrame
        """
        logger.info("Merging sentiment features with price data...")

        price_path = self.processed_path / price_file
        if not price_path.exists():
            logger.error(f"Price data not found: {price_path}")
            return pd.DataFrame()

        df_price = pd.read_csv(price_path)
        df_price['Date'] = pd.to_datetime(df_price['Date'])
        # Normalize to date-only (strip time) for merge — price dates may have
        # time components (e.g., 04:00:00 from yfinance timezone conversion)
        df_price['Date'] = df_price['Date'].dt.normalize()

        # Merge
        sentiment_cols = ['Date', 'Ticker', 'Sentiment_Index', 'Sentiment_Momentum',
                         'Article_Count', 'Positive_Ratio', 'Negative_Ratio']
        available_cols = [c for c in sentiment_cols if c in df_sentiment.columns]

        # Ensure sentiment dates are also normalized
        df_sentiment['Date'] = pd.to_datetime(df_sentiment['Date']).dt.normalize()

        df_merged = df_price.merge(
            df_sentiment[available_cols],
            on=['Date', 'Ticker'],
            how='left'
        )

        # Fill missing sentiment days with neutral values
        df_merged['Sentiment_Index'] = df_merged['Sentiment_Index'].fillna(0.0)
        df_merged['Sentiment_Momentum'] = df_merged['Sentiment_Momentum'].fillna(0.0)
        df_merged['Article_Count'] = df_merged['Article_Count'].fillna(0).astype(int)
        df_merged['Positive_Ratio'] = df_merged['Positive_Ratio'].fillna(0.0)
        df_merged['Negative_Ratio'] = df_merged['Negative_Ratio'].fillna(0.0)

        logger.info(f"Merged dataset: {len(df_merged)} rows, {len(df_merged.columns)} columns")
        logger.info(f"Days with sentiment data: {(df_merged['Article_Count'] > 0).sum()}")

        return df_merged

    def run_feature_pipeline(self,
                              scored_file: str = 'news_scored.csv',
                              price_file: str = 'daily_ohlcv_processed.csv',
                              output_file: str = 'merged_dataset.csv') -> pd.DataFrame:
        """
        Run the full sentiment feature pipeline:
        1. Load scored news
        2. Compute Daily Sentiment Index
        3. Add Sentiment Momentum
        4. Merge with price data
        5. Save merged dataset

        Returns:
            Merged DataFrame ready for LSTM modeling
        """
        logger.info("=" * 60)
        logger.info("PHASE 2b: SENTIMENT FEATURE ENGINEERING")
        logger.info("=" * 60)

        # Load scored news
        scored_path = self.processed_path / scored_file
        if not scored_path.exists():
            logger.error(f"Scored news not found: {scored_path}")
            logger.info("Run Phase 2a (sentiment scoring) first.")
            return pd.DataFrame()

        df_scored = pd.read_csv(scored_path)
        logger.info(f"Loaded {len(df_scored)} scored articles")

        # Compute daily index
        df_daily = self.compute_daily_sentiment_index(df_scored)
        if df_daily.empty:
            logger.error("Daily sentiment computation failed")
            return pd.DataFrame()

        # Add momentum
        df_daily = self.add_sentiment_momentum(df_daily)

        # Merge with price data
        df_merged = self.merge_with_price_data(df_daily, price_file)
        if df_merged.empty:
            logger.error("Merge failed")
            return pd.DataFrame()

        # Save
        output_path = self.processed_path / output_file
        df_merged.to_csv(output_path, index=False)
        logger.info(f"Saved merged dataset to {output_path}")

        logger.info("=" * 60)
        logger.info("SENTIMENT FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 60)

        return df_merged


def main():
    engineer = SentimentFeatureEngineer()
    engineer.run_feature_pipeline()


if __name__ == "__main__":
    main()
