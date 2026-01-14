"""
Data Engineering Pipeline - Summary Statistics
Generate summary statistics and visualizations for collected data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineSummary:
    """Generate summary statistics and visualizations"""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """Initialize summary generator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_path = Path(self.config['paths']['processed'])
        self.results_path = Path(self.config['paths']['results'])
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def generate_financial_summary(self):
        """Generate summary for financial data"""
        logger.info("Generating financial data summary...")
        
        # Load data
        daily_file = self.processed_path / 'daily_ohlcv_processed.csv'
        if not daily_file.exists():
            logger.warning("Daily data not found")
            return
        
        df = pd.read_csv(daily_file)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Summary statistics
        summary = {
            'total_records': len(df),
            'tickers': df['Ticker'].unique().tolist(),
            'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
            'days_covered': (df['Date'].max() - df['Date'].min()).days,
            'features': len(df.columns)
        }
        
        logger.info("Financial Data Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        # Create visualizations
        self._plot_price_comparison(df)
        self._plot_volume_analysis(df)
        self._plot_volatility_comparison(df)
        
        return summary
    
    def generate_textual_summary(self):
        """Generate summary for textual data"""
        logger.info("Generating textual data summary...")
        
        # Load data
        news_file = self.processed_path / 'news_processed.csv'
        if not news_file.exists():
            logger.warning("News data not found")
            return
        
        df = pd.read_csv(news_file)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Summary statistics
        summary = {
            'total_articles': len(df),
            'sources': df['source'].value_counts().to_dict() if 'source' in df.columns else {},
            'avg_text_length': df['cleaned_text'].str.len().mean() if 'cleaned_text' in df.columns else 0,
            'avg_token_count': df['token_count'].mean() if 'token_count' in df.columns else 0
        }
        
        logger.info("Textual Data Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        # Create visualizations
        self._plot_news_timeline(df)
        self._plot_news_sources(df)
        
        return summary
    
    def _plot_price_comparison(self, df):
        """Plot price comparison across tickers"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker].sort_values('Date')
            ax.plot(ticker_data['Date'], ticker_data['Close'], label=ticker, linewidth=2)
        
        ax.set_title('Price Comparison - Core Set vs Benchmark Set', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Normalized Close Price', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'price_comparison.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {self.results_path / 'price_comparison.png'}")
        plt.close()
    
    def _plot_volume_analysis(self, df):
        """Plot volume analysis"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Volume over time
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker].sort_values('Date')
            axes[0].plot(ticker_data['Date'], ticker_data['Volume'], label=ticker, alpha=0.7)
        
        axes[0].set_title('Trading Volume Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Volume', fontsize=12)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Average volume by ticker
        avg_volume = df.groupby('Ticker')['Volume'].mean().sort_values(ascending=False)
        axes[1].bar(avg_volume.index, avg_volume.values)
        axes[1].set_title('Average Trading Volume by Ticker', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Ticker', fontsize=12)
        axes[1].set_ylabel('Average Volume', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'volume_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {self.results_path / 'volume_analysis.png'}")
        plt.close()
    
    def _plot_volatility_comparison(self, df):
        """Plot volatility comparison"""
        if 'Volatility_20d' not in df.columns:
            logger.warning("Volatility data not found, skipping visualization")
            return
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker].sort_values('Date')
            ax.plot(ticker_data['Date'], ticker_data['Volatility_20d'], label=ticker, linewidth=2)
        
        ax.set_title('20-Day Volatility Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Volatility (20-day)', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'volatility_comparison.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {self.results_path / 'volatility_comparison.png'}")
        plt.close()
    
    def _plot_news_timeline(self, df):
        """Plot news article timeline"""
        if 'date' not in df.columns:
            logger.warning("Date column not found in news data")
            return
        
        df_clean = df.dropna(subset=['date'])
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        news_by_date = df_clean.groupby(df_clean['date'].dt.date).size()
        
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(news_by_date.index, news_by_date.values, linewidth=2)
        ax.fill_between(news_by_date.index, news_by_date.values, alpha=0.3)
        
        ax.set_title('News Article Volume Over Time', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Number of Articles', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'news_timeline.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {self.results_path / 'news_timeline.png'}")
        plt.close()
    
    def _plot_news_sources(self, df):
        """Plot news sources distribution"""
        if 'source' not in df.columns:
            logger.warning("Source column not found in news data")
            return
        
        source_counts = df['source'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(source_counts.index, source_counts.values)
        ax.set_title('News Articles by Source', fontsize=16, fontweight='bold')
        ax.set_xlabel('Source', fontsize=12)
        ax.set_ylabel('Number of Articles', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'news_sources.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {self.results_path / 'news_sources.png'}")
        plt.close()
    
    def generate_full_report(self):
        """Generate complete summary report"""
        logger.info("=" * 70)
        logger.info("GENERATING PIPELINE SUMMARY REPORT")
        logger.info("=" * 70)
        
        # Generate summaries
        fin_summary = self.generate_financial_summary()
        text_summary = self.generate_textual_summary()
        
        # Create comprehensive report
        report = {
            'generated_at': datetime.now().isoformat(),
            'financial_data': fin_summary,
            'textual_data': text_summary
        }
        
        # Save report as YAML
        report_file = self.results_path / 'pipeline_summary.yaml'
        with open(report_file, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        logger.info(f"\n✓ Summary report saved to: {report_file}")
        logger.info("=" * 70)
        logger.info("SUMMARY GENERATION COMPLETE")
        logger.info("=" * 70)


def main():
    """Main execution"""
    summary = PipelineSummary()
    summary.generate_full_report()


if __name__ == "__main__":
    main()
