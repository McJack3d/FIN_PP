import pandas as pd
import logging

logger = logging.getLogger(__name__)

def validate_stock_data(df, ticker=None):
    """Validate stock data quality"""
    issues = {}
    
    if df is None or df.empty:
        issues["empty"] = "DataFrame is empty or None"
        return False, issues
    
    # Check for required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues["missing_columns"] = f"Missing required columns: {missing_cols}"
    
    # Check for null values
    null_counts = df[list(set(required_cols) & set(df.columns))].isnull().sum()
    if null_counts.sum() > 0:
        null_cols = null_counts[null_counts > 0].to_dict()
        issues["null_values"] = f"Found null values: {null_cols}"
    
    # Check for negative prices
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            neg_count = (df[col] <= 0).sum()
            if neg_count > 0:
                issues[f"negative_{col}"] = f"Found {neg_count} negative or zero values in {col}"
    
    # Decide if data is valid overall
    is_valid = len(issues) == 0
    
    return is_valid, issues

def validate_news_data(data):
    """Validate news data quality"""
    issues = {}
    
    # Handle different input formats
    if isinstance(data, dict):
        # API response format
        if 'status' in data and data['status'] != 'ok':
            issues["api_error"] = f"API error: {data.get('status')} - {data.get('message')}"
            return False, issues
        
        if 'articles' not in data:
            issues["missing_articles"] = "No 'articles' key in response"
            return False, issues
    
    elif isinstance(data, pd.DataFrame):
        # DataFrame format
        if data.empty:
            issues["empty"] = "DataFrame is empty"
            return False, issues
            
        # Check required columns
        required_cols = ['title', 'publishedAt']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues["missing_columns"] = f"Missing required columns: {missing_cols}"
    
    # Decide if data is valid overall
    is_valid = len(issues) == 0
    
    return is_valid, issues