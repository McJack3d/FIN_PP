import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import logging
from config import PRO, RAW

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visualize.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def plot_stock_price_with_sentiment(ticker, save=True):
    """
    Plot stock price with sentiment overlay
    
    Args:
        ticker (str): Ticker symbol
        save (bool): Whether to save the figure
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    clean_ticker = ticker.replace('.', '_')
    
    # Path to merged price and sentiment data
    data_path = PRO / f"{clean_ticker}_prices_sentiment.csv"
    
    if not data_path.exists():
        logger.error(f"Merged data file not found for {ticker}")
        return None
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    
    # Check if sentiment data exists
    has_sentiment = 'sentiment_mean' in df.columns
    has_finbert = 'finbert_mean' in df.columns
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot stock price
    ax1.plot(df.index, df['Close'], 'b-', label='Stock Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Add sentiment overlay if available
    if has_sentiment or has_finbert:
        ax2 = ax1.twinx()
        
        if has_sentiment:
            ax2.plot(df.index, df['sentiment_mean'], 'r-', alpha=0.7, label='VADER Sentiment')
        
        if has_finbert:
            ax2.plot(df.index, df['finbert_mean'], 'g-', alpha=0.7, label='FinBERT Sentiment')
        
        ax2.set_ylabel('Sentiment', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add title and legend
    plt.title(f'{ticker} Stock Price and News Sentiment')
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.tight_layout()
    
    # Save figure if requested
    if save:
        results_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
        results_dir.mkdir(exist_ok=True, parents=True)
        fig_path = results_dir / f"{clean_ticker}_price_sentiment.png"
        plt.savefig(fig_path)
        logger.info(f"Saved figure to {fig_path}")
    
    return fig

def plot_sentiment_correlation(ticker, save=True):
    """
    Plot correlation between sentiment and price changes
    
    Args:
        ticker (str): Ticker symbol
        save (bool): Whether to save the figure
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    clean_ticker = ticker.replace('.', '_')
    
    # Path to merged price and sentiment data
    data_path = PRO / f"{clean_ticker}_prices_sentiment.csv"
    
    if not data_path.exists():
        logger.error(f"Merged data file not found for {ticker}")
        return None
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=['date'])
    
    # Check if required columns exist
    required_cols = ['daily_change', 'sentiment_mean']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Required columns not found in data")
        return None
    
    # Filter out rows with missing values
    df_filtered = df.dropna(subset=required_cols)
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_filtered['sentiment_mean'], df_filtered['daily_change'], alpha=0.6)
    
    # Add trend line
    z = np.polyfit(df_filtered['sentiment_mean'], df_filtered['daily_change'], 1)
    p = np.poly1d(z)
    ax.plot(df_filtered['sentiment_mean'], p(df_filtered['sentiment_mean']), "r--", alpha=0.8)
    
    # Add correlation coefficient
    corr = df_filtered['sentiment_mean'].corr(df_filtered['daily_change'])
    ax.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction')
    
    # Add labels and title
    ax.set_xlabel('News Sentiment')
    ax.set_ylabel('Daily Price Change (%)')
    ax.set_title(f'{ticker} - Correlation between News Sentiment and Price Changes')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # Save figure if requested
    if save:
        results_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
        results_dir.mkdir(exist_ok=True, parents=True)
        fig_path = results_dir / f"{clean_ticker}_sentiment_correlation.png"
        plt.savefig(fig_path)
        logger.info(f"Saved figure to {fig_path}")
    
    return fig

def plot_feature_importance(ticker, model_path=None, save=True):
    """
    Plot feature importance from a trained model
    
    Args:
        ticker (str): Ticker symbol
        model_path (str): Path to saved model
        save (bool): Whether to save the figure
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    if model_path is None:
        # Find latest model for this ticker
        models_dir = Path(__file__).resolve().parents[1] / "models"
        clean_ticker = ticker.replace('.', '_')
        model_files = list(models_dir.glob(f"{clean_ticker}_*_model_*.joblib"))
        
        if not model_files:
            logger.error(f"No model files found for {ticker}")
            return None
        
        # Get the most recent model
        model_path = max(model_files, key=lambda x: x.stat().st_mtime)
    
    # Load model data
    try:
        model_data = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
    
    # Check if model has feature importances
    model = model_data.get('model')
    features = model_data.get('features')
    
    if model is None or features is None or not hasattr(model, 'feature_importances_'):
        logger.error("Model doesn't have feature importance information")
        return None
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot feature importances
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'{ticker} - Feature Importance')
    
    # Save figure if requested
    if save:
        results_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
        results_dir.mkdir(exist_ok=True, parents=True)
        fig_path = results_dir / f"{clean_ticker}_feature_importance.png"
        plt.savefig(fig_path)
        logger.info(f"Saved figure to {fig_path}")
    
    return fig

def create_interactive_dashboard(ticker, save=True):
    """
    Create an interactive dashboard with Plotly
    
    Args:
        ticker (str): Ticker symbol
        save (bool): Whether to save the dashboard
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    clean_ticker = ticker.replace('.', '_')
    
    # Path to merged price and sentiment data
    data_path = PRO / f"{clean_ticker}_prices_sentiment.csv"
    
    if not data_path.exists():
        logger.error(f"Merged data file not found for {ticker}")
        return None
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=['date'])
    
    # Create subplots
    fig = make_subplots(rows=3, cols=1,
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       subplot_titles=(f"{ticker} Stock Price", "News Sentiment", "Trading Volume"))
    
    # Add stock price
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['Close'], name='Stock Price'),
        row=1, col=1
    )
    
    # Add sentiment if available
    if 'sentiment_mean' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['sentiment_mean'], name='VADER Sentiment',
                      line=dict(color='red')),
            row=2, col=1
        )
    
    if 'finbert_mean' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['finbert_mean'], name='FinBERT Sentiment',
                      line=dict(color='green')),
            row=2, col=1
        )
    
    # Add volume
    if 'Volume' in df.columns:
        fig.add_trace(
            go.Bar(x=df['date'], y=df['Volume'], name='Volume'),
            row=3, col=1
        )
    
    # Add news article count if available
    if 'article_count' in df.columns:
        # Create a secondary y-axis for article count
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['article_count'], name='News Articles',
                      line=dict(color='purple', width=1, dash='dot')),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=900,
        width=1200,
        title_text=f"{ticker} Stock Price, Sentiment, and Volume Analysis",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add zero line to sentiment plot
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='gray', row=2, col=1)
    
    # Save figure if requested
    if save:
        results_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
        results_dir.mkdir(exist_ok=True, parents=True)
        html_path = results_dir / f"{clean_ticker}_dashboard.html"
        fig.write_html(str(html_path))
        logger.info(f"Saved interactive dashboard to {html_path}")
    
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create visualizations for sentiment analysis")
    parser.add_argument("--ticker", required=True, help="Ticker symbol to visualize")
    parser.add_argument("--all", action="store_true", help="Create all visualizations")
    parser.add_argument("--price", action="store_true", help="Create price and sentiment plot")
    parser.add_argument("--corr", action="store_true", help="Create sentiment correlation plot")
    parser.add_argument("--importance", action="store_true", help="Create feature importance plot")
    parser.add_argument("--dashboard", action="store_true", help="Create interactive dashboard")
    
    args = parser.parse_args()
    
    # Determine which visualizations to create
    create_all = args.all or not any([args.price, args.corr, args.importance, args.dashboard])
    
    if args.price or create_all:
        plot_stock_price_with_sentiment(args.ticker)
    
    if args.corr or create_all:
        plot_sentiment_correlation(args.ticker)
    
    if args.importance or create_all:
        plot_feature_importance(args.ticker)
    
    if args.dashboard or create_all:
        create_interactive_dashboard(args.ticker)