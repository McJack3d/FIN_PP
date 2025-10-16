import logging
from src.fetch import fetch_all_data
from src.preprocess import clean_stock_data, process_news_data, merge_data_sources
from src.sentiment import analyze_news_sentiment, merge_sentiment_with_prices
from src.models import train_and_evaluate
from src.visualize import create_interactive_dashboard
from src.validate import validate_stock_data, validate_news_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_full_pipeline(ticker="XOM"):
    """Run complete analysis pipeline for a ticker"""
    try:
        # 1. Fetch data
        logger.info(f"Step 1: Fetching data for {ticker}")
        fetch_all_data([ticker])
        
        # 2. Preprocess data
        logger.info(f"Step 2: Preprocessing data")
        stock_df = clean_stock_data(ticker)
        
        # Validate stock data
        is_valid, issues = validate_stock_data(stock_df, ticker)
        if not is_valid:
            logger.warning(f"Stock data validation failed: {issues}")
            if "empty" in issues:
                logger.error("Cannot continue without stock data")
                return None
        
        # Continue with processing
        news_df = process_news_data(ticker)
        
        # Validate news data
        is_valid, issues = validate_news_data(news_df)
        if not is_valid:
            logger.warning(f"News data validation issues: {issues}")
        
        # Continue with pipeline
        merged_df = merge_data_sources(ticker, stock_df, news_df)
        
        # 3. Sentiment analysis
        logger.info(f"Step 3: Analyzing sentiment")
        sentiment_df = analyze_news_sentiment(ticker)
        merged_sentiment = merge_sentiment_with_prices(ticker)
        
        # 4. Model training and evaluation
        logger.info(f"Step 4: Training models")
        results = train_and_evaluate(ticker)
        
        # 5. Visualization
        logger.info(f"Step 5: Creating visualizations")
        dashboard = create_interactive_dashboard(ticker)
        
        logger.info(f"Pipeline completed successfully for {ticker}")
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="XOM", help="Ticker to analyze")
    args = parser.parse_args()
    
    run_full_pipeline(args.ticker)