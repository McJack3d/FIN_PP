"""
Data Preprocessing Module
Handles cleaning, normalization, and transformation of financial and textual data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler
import pickle
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialDataPreprocessor:
    """Preprocesses financial time series data with Min-Max scaling"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize preprocessor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_path = Path(self.config['paths']['raw'])
        self.processed_path = Path(self.config['paths']['processed'])
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        self.scalers = {}
        logger.info("Initialized Financial Data Preprocessor")
    
    def load_raw_data(self, filename: str) -> pd.DataFrame:
        """Load raw financial data"""
        filepath = self.raw_path / filename
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    
    def clean_financial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean financial data by handling missing values and outliers
        
        Args:
            df: Raw financial DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning financial data...")
        df_clean = df.copy()
        
        # Remove rows with missing critical values
        critical_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        before_count = len(df_clean)
        df_clean = df_clean.dropna(subset=critical_cols)
        after_count = len(df_clean)
        
        if before_count != after_count:
            logger.info(f"Removed {before_count - after_count} rows with missing values")
        
        # Handle zero volume
        df_clean = df_clean[df_clean['Volume'] > 0]
        
        # Detect and handle price anomalies
        for ticker in df_clean['Ticker'].unique():
            mask = df_clean['Ticker'] == ticker
            ticker_data = df_clean.loc[mask, 'Close']
            
            # Remove extreme outliers (beyond 5 standard deviations)
            mean = ticker_data.mean()
            std = ticker_data.std()
            outliers = (ticker_data > mean + 5 * std) | (ticker_data < mean - 5 * std)
            
            if outliers.any():
                logger.warning(f"Found {outliers.sum()} outliers for {ticker}")
                df_clean.loc[mask & outliers, critical_cols] = np.nan
                df_clean = df_clean.dropna(subset=critical_cols)
        
        # Ensure chronological order
        df_clean = df_clean.sort_values(['Ticker', 'Date'])
        
        logger.info(f"Cleaned data: {len(df_clean)} records remaining")
        return df_clean
    
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators and derived features
        
        Args:
            df: Cleaned financial DataFrame
            
        Returns:
            DataFrame with additional features
        """
        logger.info("Adding technical features...")
        df_features = df.copy()
        
        for ticker in df_features['Ticker'].unique():
            mask = df_features['Ticker'] == ticker
            ticker_df = df_features[mask].sort_values('Date')
            
            # Price-based features
            df_features.loc[mask, 'Daily_Return'] = ticker_df['Close'].pct_change()
            df_features.loc[mask, 'Log_Return'] = np.log(ticker_df['Close'] / ticker_df['Close'].shift(1))
            df_features.loc[mask, 'Price_Range'] = ticker_df['High'] - ticker_df['Low']
            df_features.loc[mask, 'Price_Change'] = ticker_df['Close'] - ticker_df['Open']
            
            # Moving averages
            for window in [5, 10, 20]:
                df_features.loc[mask, f'MA_{window}'] = ticker_df['Close'].rolling(window=window).mean()
                df_features.loc[mask, f'Volume_MA_{window}'] = ticker_df['Volume'].rolling(window=window).mean()
            
            # Volatility
            df_features.loc[mask, 'Volatility_5d'] = ticker_df['Daily_Return'].rolling(window=5).std()
            df_features.loc[mask, 'Volatility_20d'] = ticker_df['Daily_Return'].rolling(window=20).std()
            
            # RSI (Relative Strength Index)
            delta = ticker_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_features.loc[mask, 'RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = ticker_df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = ticker_df['Close'].ewm(span=26, adjust=False).mean()
            df_features.loc[mask, 'MACD'] = exp1 - exp2
            df_features.loc[mask, 'Signal_Line'] = df_features.loc[mask, 'MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            sma20 = ticker_df['Close'].rolling(window=20).mean()
            std20 = ticker_df['Close'].rolling(window=20).std()
            df_features.loc[mask, 'BB_Upper'] = sma20 + (std20 * 2)
            df_features.loc[mask, 'BB_Lower'] = sma20 - (std20 * 2)
            df_features.loc[mask, 'BB_Width'] = df_features.loc[mask, 'BB_Upper'] - df_features.loc[mask, 'BB_Lower']
        
        logger.info(f"Added {len(df_features.columns) - len(df.columns)} new features")
        return df_features
    
    def normalize_data(self, df: pd.DataFrame, 
                      fit: bool = True,
                      exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        Apply Min-Max scaling to numerical features
        
        Args:
            df: DataFrame to normalize
            fit: Whether to fit new scalers (True) or use existing ones (False)
            exclude_cols: Columns to exclude from scaling
            
        Returns:
            Normalized DataFrame
        """
        logger.info("Applying Min-Max normalization...")
        
        if exclude_cols is None:
            exclude_cols = ['Date', 'Ticker']
        
        df_normalized = df.copy()
        
        # Identify numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        for ticker in df['Ticker'].unique():
            mask = df_normalized['Ticker'] == ticker
            
            if fit:
                # Create and fit new scaler for this ticker
                scaler = MinMaxScaler(feature_range=(0, 1))
                df_normalized.loc[mask, numerical_cols] = scaler.fit_transform(
                    df.loc[mask, numerical_cols]
                )
                self.scalers[ticker] = scaler
            else:
                # Use existing scaler
                if ticker in self.scalers:
                    df_normalized.loc[mask, numerical_cols] = self.scalers[ticker].transform(
                        df.loc[mask, numerical_cols]
                    )
                else:
                    logger.warning(f"No scaler found for {ticker}, skipping normalization")
        
        logger.info(f"Normalized {len(numerical_cols)} numerical columns")
        return df_normalized
    
    def save_scalers(self, filename: str = 'scalers.pkl'):
        """Save fitted scalers to disk"""
        filepath = self.processed_path / filename
        with open(filepath, 'wb') as f:
            pickle.dump(self.scalers, f)
        logger.info(f"Saved scalers to {filepath}")
    
    def load_scalers(self, filename: str = 'scalers.pkl'):
        """Load scalers from disk"""
        filepath = self.processed_path / filename
        with open(filepath, 'rb') as f:
            self.scalers = pickle.load(f)
        logger.info(f"Loaded scalers from {filepath}")
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to CSV"""
        filepath = self.processed_path / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")
    
    def process_pipeline(self, input_filename: str, output_filename: str):
        """
        Complete preprocessing pipeline
        
        Args:
            input_filename: Name of raw data file
            output_filename: Name for processed data file
        """
        logger.info("=" * 60)
        logger.info("Starting financial data preprocessing pipeline")
        logger.info("=" * 60)
        
        # Load raw data
        df = self.load_raw_data(input_filename)
        logger.info(f"Loaded {len(df)} raw records")
        
        # Clean data
        df_clean = self.clean_financial_data(df)
        
        # Add technical features
        df_features = self.add_technical_features(df_clean)
        
        # Handle remaining NaN values (from rolling calculations)
        df_features = df_features.dropna()
        logger.info(f"After feature engineering: {len(df_features)} records")
        
        # Normalize data
        df_normalized = self.normalize_data(df_features, fit=True)
        
        # Save processed data and scalers
        self.save_processed_data(df_normalized, output_filename)
        self.save_scalers()
        
        logger.info("=" * 60)
        logger.info("Financial preprocessing complete!")
        logger.info("=" * 60)
        
        return df_normalized


class TextualDataPreprocessor:
    """Preprocesses textual data for FinBERT"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize text preprocessor"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.news_path = Path(self.config['paths']['news'])
        self.processed_path = Path(self.config['paths']['processed'])
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized Textual Data Preprocessor")
    
    def load_news_data(self, filename: str) -> pd.DataFrame:
        """Load news data"""
        filepath = self.news_path / filename
        logger.info(f"Loading news data from {filepath}")
        df = pd.read_csv(filepath)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    
    def clean_text(self, text: str) -> str:
        """
        Clean individual text while preserving context for FinBERT
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep punctuation (important for FinBERT)
        text = re.sub(r'[^\w\s.,!?;:\-\(\)]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize_for_finbert(self, df: pd.DataFrame, 
                            text_column: str = 'text',
                            max_length: int = 512) -> pd.DataFrame:
        """
        Tokenize text data for FinBERT
        Note: Keeps stop words as they provide important context for financial sentiment
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text
            max_length: Maximum token length (FinBERT limit is 512)
            
        Returns:
            DataFrame with tokenized text
        """
        logger.info("Tokenizing text for FinBERT...")
        
        try:
            from transformers import AutoTokenizer
            
            # Load FinBERT tokenizer
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            
            df_tokenized = df.copy()
            
            # Clean text first
            df_tokenized['cleaned_text'] = df_tokenized[text_column].apply(self.clean_text)
            
            # Tokenize
            def tokenize_text(text):
                if not text:
                    return []
                
                tokens = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors=None
                )
                return tokens
            
            df_tokenized['tokens'] = df_tokenized['cleaned_text'].apply(tokenize_text)
            df_tokenized['token_count'] = df_tokenized['tokens'].apply(
                lambda x: len(x['input_ids']) if x else 0
            )
            
            logger.info(f"Tokenized {len(df_tokenized)} texts")
            logger.info(f"Average token count: {df_tokenized['token_count'].mean():.1f}")
            
            return df_tokenized
            
        except ImportError:
            logger.warning("transformers library not installed. Skipping tokenization.")
            logger.info("Install with: pip install transformers")
            
            # Fallback: simple word tokenization
            df_tokenized = df.copy()
            df_tokenized['cleaned_text'] = df_tokenized[text_column].apply(self.clean_text)
            df_tokenized['tokens'] = df_tokenized['cleaned_text'].apply(lambda x: x.split())
            df_tokenized['token_count'] = df_tokenized['tokens'].apply(len)
            
            logger.info("Used simple word tokenization as fallback")
            return df_tokenized
    
    def filter_relevant_news(self, df: pd.DataFrame, 
                            tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filter news to only include relevant articles
        
        Args:
            df: DataFrame with news data
            tickers: List of ticker symbols to filter for
            
        Returns:
            Filtered DataFrame
        """
        if tickers is None:
            tickers = self.config['tickers']['all_tickers']
        
        logger.info(f"Filtering news for {len(tickers)} tickers...")
        
        # Create a pattern for matching tickers and related keywords
        ticker_patterns = '|'.join([re.escape(t) for t in tickers])
        
        # Additional keywords for nuclear energy context
        keywords = ['nuclear', 'SMR', 'reactor', 'uranium', 'enrichment', 
                   'energy', 'power', 'electricity', 'clean energy']
        keyword_pattern = '|'.join(keywords)
        
        combined_pattern = f'({ticker_patterns})|({keyword_pattern})'
        
        # Filter based on text content
        if 'text' in df.columns:
            df['is_relevant'] = df['text'].str.contains(
                combined_pattern, 
                case=False, 
                na=False, 
                regex=True
            )
            df_filtered = df[df['is_relevant']].copy()
            df_filtered = df_filtered.drop('is_relevant', axis=1)
        else:
            df_filtered = df.copy()
        
        logger.info(f"Filtered to {len(df_filtered)} relevant articles ({len(df_filtered)/len(df)*100:.1f}%)")
        return df_filtered
    
    def deduplicate_news(self, df: pd.DataFrame, 
                        text_column: str = 'text') -> pd.DataFrame:
        """
        Remove duplicate news articles
        
        Args:
            df: DataFrame with news data
            text_column: Column to check for duplicates
            
        Returns:
            Deduplicated DataFrame
        """
        logger.info("Removing duplicate articles...")
        before_count = len(df)
        
        # Remove exact duplicates
        df_dedup = df.drop_duplicates(subset=[text_column], keep='first')
        
        after_count = len(df_dedup)
        logger.info(f"Removed {before_count - after_count} duplicate articles")
        
        return df_dedup
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed text data"""
        filepath = self.processed_path / filename
        
        # If tokens column exists and contains dicts, convert to string for CSV
        if 'tokens' in df.columns:
            df_save = df.copy()
            df_save['tokens'] = df_save['tokens'].astype(str)
            df_save.to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
        
        logger.info(f"Saved processed text data to {filepath}")
    
    def process_pipeline(self, input_filename: str, output_filename: str):
        """
        Complete text preprocessing pipeline
        
        Args:
            input_filename: Name of raw news file
            output_filename: Name for processed news file
        """
        logger.info("=" * 60)
        logger.info("Starting textual data preprocessing pipeline")
        logger.info("=" * 60)
        
        # Load news data
        df = self.load_news_data(input_filename)
        logger.info(f"Loaded {len(df)} news articles")
        
        # Deduplicate
        df_dedup = self.deduplicate_news(df)
        
        # Filter relevant news
        df_filtered = self.filter_relevant_news(df_dedup)
        
        # Tokenize for FinBERT
        df_tokenized = self.tokenize_for_finbert(df_filtered)
        
        # Save processed data
        self.save_processed_data(df_tokenized, output_filename)
        
        logger.info("=" * 60)
        logger.info("Textual preprocessing complete!")
        logger.info("=" * 60)
        
        return df_tokenized


def main():
    """Main execution function"""
    # Process financial data
    logger.info("\n" + "=" * 60)
    logger.info("FINANCIAL DATA PREPROCESSING")
    logger.info("=" * 60)
    
    fin_processor = FinancialDataPreprocessor()
    fin_processor.process_pipeline('daily_ohlcv.csv', 'daily_ohlcv_processed.csv')
    
    # Process textual data
    logger.info("\n" + "=" * 60)
    logger.info("TEXTUAL DATA PREPROCESSING")
    logger.info("=" * 60)
    
    text_processor = TextualDataPreprocessor()
    text_processor.process_pipeline('all_news_combined.csv', 'news_processed.csv')


if __name__ == "__main__":
    main()
