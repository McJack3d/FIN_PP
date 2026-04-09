"""
Thesis Visualizations (Phase 3)
Generates all figures needed for the thesis defense:
  - Prediction vs Actual time series per ticker
  - Training loss curves
  - Sentiment coverage & distribution analysis
  - Sentiment-return correlation heatmaps
  - Model comparison bar charts
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThesisVisualizations:
    """Generates all thesis-ready figures from pipeline outputs."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.results_path = Path(self.config['paths']['results'])
        self.processed_path = Path(self.config['paths']['processed'])
        self.results_path.mkdir(parents=True, exist_ok=True)

        self.core_set = self.config['tickers']['core_set']
        self.benchmark_set = self.config['tickers']['benchmark_set']
        self.all_tickers = self.config['tickers']['all_tickers']

        # Consistent color palette
        self.COLORS = {
            'baseline': '#3498db',
            'augmented': '#e74c3c',
            'actual': '#2c3e50',
            'core': '#e74c3c',
            'benchmark': '#3498db',
            'positive': '#27ae60',
            'negative': '#e74c3c',
            'neutral': '#95a5a6',
        }

        logger.info("ThesisVisualizations initialized")

    # ==================================================================
    # 1. PREDICTION vs ACTUAL PLOTS
    # ==================================================================
    def plot_predictions_vs_actual(self, target: str = 'Forward_Return_1d'):
        """
        Time-series plot of predicted vs actual returns per ticker.
        Shows both baseline and augmented model predictions overlaid on actuals.
        """
        pred_file = self.results_path / f'predictions_{target}.csv'
        if not pred_file.exists():
            logger.warning(f"No predictions file: {pred_file}")
            return

        df = pd.read_csv(pred_file)
        df['Date'] = pd.to_datetime(df['Date'])

        tickers = df['Ticker'].unique()
        n_tickers = len(tickers)
        n_cols = 3
        n_rows = (n_tickers + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows),
                                 sharex=False)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_tickers == 1 else axes.flatten()

        for idx, ticker in enumerate(sorted(tickers)):
            ax = axes[idx]
            ticker_data = df[df['Ticker'] == ticker].sort_values('Date')

            # Baseline predictions
            base = ticker_data[ticker_data['Model'] == 'baseline']
            aug = ticker_data[ticker_data['Model'] == 'augmented']

            if not base.empty:
                ax.plot(base['Date'], base['y_actual'], color=self.COLORS['actual'],
                        linewidth=1, alpha=0.8, label='Actual')
                ax.plot(base['Date'], base['y_pred'], color=self.COLORS['baseline'],
                        linewidth=0.8, alpha=0.7, label='Baseline LSTM')
            if not aug.empty:
                ax.plot(aug['Date'], aug['y_pred'], color=self.COLORS['augmented'],
                        linewidth=0.8, alpha=0.7, label='Sentiment LSTM')

            group = 'Core' if ticker in self.core_set else 'Bench'
            ax.set_title(f'{ticker} ({group})', fontsize=11, fontweight='bold')
            ax.set_ylabel('Return', fontsize=9)
            ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
            ax.legend(fontsize=7, loc='upper right')
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Hide unused subplots
        for idx in range(n_tickers, len(axes)):
            axes[idx].set_visible(False)

        horizon = '1-day' if '1d' in target else '5-day'
        fig.suptitle(f'Predicted vs Actual {horizon} Forward Returns',
                     fontsize=16, fontweight='bold', y=1.01)
        plt.tight_layout()
        path = self.results_path / f'predictions_vs_actual_{target}.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved: {path}")

    # ==================================================================
    # 2. TRAINING LOSS CURVES
    # ==================================================================
    def plot_training_curves(self, experiment_results: dict):
        """
        Plot training and validation loss curves for all models.
        One subplot per ticker, baseline vs augmented overlaid.
        """
        for target_col, target_results in experiment_results.items():
            if not target_results:
                continue

            tickers = sorted(target_results.keys())
            n = len(tickers)
            n_cols = 3
            n_rows = (n + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
            axes = axes.flatten() if n_rows > 1 else (
                [axes] if n == 1 else axes.flatten()
            )

            for idx, ticker in enumerate(tickers):
                ax = axes[idx]
                data = target_results[ticker]

                for model_type, color, label in [
                    ('baseline', self.COLORS['baseline'], 'Baseline'),
                    ('augmented', self.COLORS['augmented'], 'Sentiment'),
                ]:
                    result = data.get(model_type, {})
                    if not result or 'history' not in result:
                        continue
                    history = result['history']
                    epochs = range(1, len(history['loss']) + 1)
                    ax.plot(epochs, history['loss'], color=color,
                            linewidth=1, alpha=0.8, label=f'{label} Train')
                    if 'val_loss' in history:
                        ax.plot(epochs, history['val_loss'], color=color,
                                linewidth=1, linestyle='--', alpha=0.6,
                                label=f'{label} Val')

                group = 'Core' if ticker in self.core_set else 'Bench'
                ax.set_title(f'{ticker} ({group})', fontsize=11, fontweight='bold')
                ax.set_xlabel('Epoch', fontsize=9)
                ax.set_ylabel('MSE Loss', fontsize=9)
                ax.legend(fontsize=7)
                ax.tick_params(labelsize=8)

            for idx in range(n, len(axes)):
                axes[idx].set_visible(False)

            horizon = '1-day' if '1d' in target_col else '5-day'
            fig.suptitle(f'Training Curves — {horizon} Forward Return',
                         fontsize=16, fontweight='bold', y=1.01)
            plt.tight_layout()
            path = self.results_path / f'training_curves_{target_col}.png'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved: {path}")

    # ==================================================================
    # 3. SENTIMENT COVERAGE & DISTRIBUTION
    # ==================================================================
    def plot_sentiment_analysis(self):
        """
        Comprehensive sentiment analysis figures:
        a) Coverage heatmap: % days with news per ticker per month
        b) Sentiment score distribution by ticker
        c) Article count per ticker (bar chart)
        """
        merged_file = self.processed_path / 'merged_dataset.csv'
        scored_file = self.processed_path / 'news_scored.csv'

        if not merged_file.exists():
            logger.warning("No merged dataset found for sentiment analysis")
            return

        df = pd.read_csv(merged_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df['YearMonth'] = df['Date'].dt.to_period('M')

        # --- (a) Sentiment coverage heatmap ---
        fig, ax = plt.subplots(figsize=(14, 6))
        pivot = df.pivot_table(
            values='Article_Count',
            index='Ticker',
            columns='YearMonth',
            aggfunc=lambda x: (x > 0).mean() * 100
        )
        # Sort: core set first
        order = [t for t in self.core_set if t in pivot.index] + \
                [t for t in self.benchmark_set if t in pivot.index]
        pivot = pivot.reindex(order)

        im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)
        x_labels = [str(p) for p in pivot.columns]
        step = max(1, len(x_labels) // 12)
        ax.set_xticks(range(0, len(x_labels), step))
        ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), step)],
                           fontsize=8, rotation=45)
        ax.set_title('Sentiment Coverage: % of Trading Days with News per Month',
                     fontsize=14, fontweight='bold')
        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val) and val > 0:
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                            fontsize=6, color='black' if val < 50 else 'white')
        # Add separator line between core and benchmark
        n_core = len([t for t in self.core_set if t in pivot.index])
        if n_core < len(pivot.index):
            ax.axhline(y=n_core - 0.5, color='white', linewidth=2)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('% days with news', fontsize=10)
        plt.tight_layout()
        path = self.results_path / 'sentiment_coverage_heatmap.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved: {path}")

        # --- (b) Sentiment score distribution by ticker ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Box plot of Sentiment_Index by ticker
        tickers_ordered = [t for t in self.core_set if t in df['Ticker'].unique()] + \
                          [t for t in self.benchmark_set if t in df['Ticker'].unique()]
        data_for_box = [df[(df['Ticker'] == t) & (df['Sentiment_Index'] != 0)]['Sentiment_Index'].values
                        for t in tickers_ordered]
        # Filter out empty arrays
        valid_tickers = []
        valid_data = []
        for t, d in zip(tickers_ordered, data_for_box):
            if len(d) > 0:
                valid_tickers.append(t)
                valid_data.append(d)

        if valid_data:
            bp = axes[0].boxplot(valid_data, labels=valid_tickers, patch_artist=True)
            n_core_valid = len([t for t in self.core_set if t in valid_tickers])
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(self.COLORS['core'] if i < n_core_valid else self.COLORS['benchmark'])
                patch.set_alpha(0.6)
            axes[0].axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
            axes[0].set_title('Sentiment Index Distribution (non-zero days only)',
                             fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Sentiment Index')
            axes[0].tick_params(labelsize=9)

        # Article count bar chart
        article_counts = df.groupby('Ticker')['Article_Count'].apply(lambda x: (x > 0).sum())
        article_counts = article_counts.reindex(tickers_ordered)
        colors = [self.COLORS['core'] if t in self.core_set else self.COLORS['benchmark']
                  for t in tickers_ordered if t in article_counts.index]
        bars = axes[1].bar(range(len(article_counts)), article_counts.values, color=colors, alpha=0.7)
        axes[1].set_xticks(range(len(article_counts)))
        axes[1].set_xticklabels(article_counts.index, fontsize=9)
        axes[1].set_title('Trading Days with News Coverage per Ticker',
                         fontsize=12, fontweight='bold')
        axes[1].set_ylabel('# Days with at Least 1 Article')
        axes[1].axhline(y=article_counts.mean(), color='gray', linewidth=1, linestyle='--',
                        label=f'Mean: {article_counts.mean():.0f}')
        axes[1].legend(fontsize=9)

        # Add count labels on bars
        for bar, val in zip(bars, article_counts.values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        path = self.results_path / 'sentiment_distribution.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved: {path}")

    # ==================================================================
    # 4. SENTIMENT-RETURN CORRELATION
    # ==================================================================
    def plot_sentiment_return_correlation(self):
        """
        Scatter plots + correlation: Sentiment_Index vs Forward_Return
        for each ticker, split by core vs benchmark.
        """
        merged_file = self.processed_path / 'merged_dataset.csv'
        if not merged_file.exists():
            return

        df = pd.read_csv(merged_file)
        df = df[df['Sentiment_Index'] != 0].copy()  # Only days with sentiment

        if df.empty or len(df) < 10:
            logger.warning("Too few sentiment days for correlation analysis")
            return

        for target in ['Forward_Return_1d', 'Forward_Return_5d']:
            if target not in df.columns:
                continue

            tickers = sorted(df['Ticker'].unique())
            n = len(tickers)
            n_cols = 3
            n_rows = (n + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
            axes = axes.flatten() if n_rows > 1 else (
                [axes] if n == 1 else axes.flatten()
            )

            for idx, ticker in enumerate(tickers):
                ax = axes[idx]
                sub = df[df['Ticker'] == ticker].dropna(subset=[target, 'Sentiment_Index'])

                if len(sub) < 5:
                    ax.text(0.5, 0.5, f'{ticker}\n(n<5)', ha='center', va='center',
                            transform=ax.transAxes, fontsize=12)
                    ax.set_title(ticker, fontsize=11)
                    continue

                color = self.COLORS['core'] if ticker in self.core_set else self.COLORS['benchmark']
                ax.scatter(sub['Sentiment_Index'], sub[target],
                          alpha=0.5, s=20, color=color, edgecolors='none')

                # Regression line
                from numpy.polynomial.polynomial import polyfit
                b, m = polyfit(sub['Sentiment_Index'].values, sub[target].values, 1)
                x_line = np.linspace(sub['Sentiment_Index'].min(), sub['Sentiment_Index'].max(), 50)
                ax.plot(x_line, b + m * x_line, color='black', linewidth=1, linestyle='--')

                # Correlation
                corr = sub['Sentiment_Index'].corr(sub[target])
                group = 'Core' if ticker in self.core_set else 'Bench'
                ax.set_title(f'{ticker} ({group}) r={corr:.3f} (n={len(sub)})',
                            fontsize=10, fontweight='bold')
                ax.set_xlabel('Sentiment Index', fontsize=8)
                ax.set_ylabel(target.replace('_', ' '), fontsize=8)
                ax.axhline(y=0, color='gray', linewidth=0.3, linestyle='-')
                ax.axvline(x=0, color='gray', linewidth=0.3, linestyle='-')
                ax.tick_params(labelsize=7)

            for idx in range(n, len(axes)):
                axes[idx].set_visible(False)

            horizon = '1-day' if '1d' in target else '5-day'
            fig.suptitle(f'Sentiment Index vs {horizon} Forward Return (news days only)',
                         fontsize=14, fontweight='bold', y=1.01)
            plt.tight_layout()
            path = self.results_path / f'sentiment_return_scatter_{target}.png'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved: {path}")

    # ==================================================================
    # 5. MODEL COMPARISON BAR CHARTS
    # ==================================================================
    def plot_model_comparison(self):
        """
        Side-by-side bar charts comparing Baseline vs Augmented:
        - MAE per ticker (grouped bar)
        - Directional Accuracy per ticker (grouped bar)
        - DM test significance markers
        """
        for target in self.config['model']['target_columns']:
            eval_file = self.results_path / f'evaluation_{target}.csv'
            if not eval_file.exists():
                continue

            df = pd.read_csv(eval_file)

            # Sort: core first, then benchmark
            order = [t for t in self.core_set if t in df['Ticker'].values] + \
                    [t for t in self.benchmark_set if t in df['Ticker'].values]
            df = df.set_index('Ticker').reindex(order).reset_index()

            fig, axes = plt.subplots(1, 3, figsize=(20, 7))
            x = np.arange(len(df))
            width = 0.35

            # --- MAE ---
            ax = axes[0]
            bars1 = ax.bar(x - width / 2, df['MAE_Baseline'], width,
                          label='Baseline LSTM', color=self.COLORS['baseline'], alpha=0.8)
            bars2 = ax.bar(x + width / 2, df['MAE_Augmented'], width,
                          label='Sentiment LSTM', color=self.COLORS['augmented'], alpha=0.8)
            # Mark significant DM results
            for i, row in df.iterrows():
                if row['DM_Significant']:
                    better = 'augmented' if row['MAE_Augmented'] < row['MAE_Baseline'] else 'baseline'
                    y = max(row['MAE_Baseline'], row['MAE_Augmented']) * 1.05
                    ax.text(i, y, '*', ha='center', fontsize=16, fontweight='bold',
                           color=self.COLORS['augmented'] if better == 'augmented' else self.COLORS['baseline'])
            ax.set_xticks(x)
            ax.set_xticklabels(df['Ticker'], fontsize=9)
            ax.set_ylabel('MAE', fontsize=11)
            ax.set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            # Separator line
            n_core = len([t for t in self.core_set if t in df['Ticker'].values])
            if n_core < len(df):
                ax.axvline(x=n_core - 0.5, color='gray', linewidth=1, linestyle=':')
                ax.text(n_core / 2, ax.get_ylim()[1] * 0.95, 'Core', ha='center',
                       fontsize=9, color='gray', style='italic')
                ax.text((n_core + len(df)) / 2, ax.get_ylim()[1] * 0.95, 'Benchmark',
                       ha='center', fontsize=9, color='gray', style='italic')

            # --- Directional Accuracy ---
            ax = axes[1]
            ax.bar(x - width / 2, df['DA_Baseline'], width,
                  label='Baseline LSTM', color=self.COLORS['baseline'], alpha=0.8)
            ax.bar(x + width / 2, df['DA_Augmented'], width,
                  label='Sentiment LSTM', color=self.COLORS['augmented'], alpha=0.8)
            ax.axhline(y=50, color='gray', linewidth=1, linestyle='--', label='Random (50%)')
            ax.set_xticks(x)
            ax.set_xticklabels(df['Ticker'], fontsize=9)
            ax.set_ylabel('Directional Accuracy (%)', fontsize=11)
            ax.set_title('Directional Accuracy', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            if n_core < len(df):
                ax.axvline(x=n_core - 0.5, color='gray', linewidth=1, linestyle=':')

            # --- MAE Improvement % ---
            ax = axes[2]
            colors = [self.COLORS['positive'] if v > 0 else self.COLORS['negative']
                     for v in df['MAE_Improvement_Pct']]
            bars = ax.bar(x, df['MAE_Improvement_Pct'], color=colors, alpha=0.8)
            ax.axhline(y=0, color='black', linewidth=0.8)
            for i, row in df.iterrows():
                if row['DM_Significant']:
                    y = row['MAE_Improvement_Pct']
                    offset = 1.5 if y >= 0 else -1.5
                    ax.text(i, y + offset, '*', ha='center', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(df['Ticker'], fontsize=9)
            ax.set_ylabel('MAE Improvement (%)', fontsize=11)
            ax.set_title('Sentiment Impact (+ = helps)', fontsize=13, fontweight='bold')
            if n_core < len(df):
                ax.axvline(x=n_core - 0.5, color='gray', linewidth=1, linestyle=':')

            horizon = '1-day' if '1d' in target else '5-day'
            fig.suptitle(f'Model Comparison — {horizon} Forward Return (* = statistically significant)',
                         fontsize=15, fontweight='bold', y=1.02)
            plt.tight_layout()
            path = self.results_path / f'model_comparison_{target}.png'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved: {path}")

    # ==================================================================
    # MASTER: Generate all thesis figures
    # ==================================================================
    def generate_all(self, experiment_results: dict = None):
        """Generate all thesis visualizations."""
        logger.info("=" * 70)
        logger.info("GENERATING THESIS VISUALIZATIONS")
        logger.info("=" * 70)

        # Prediction vs actual
        for target in self.config['model']['target_columns']:
            self.plot_predictions_vs_actual(target)

        # Training curves (need experiment_results with history)
        if experiment_results:
            self.plot_training_curves(experiment_results)

        # Sentiment analysis
        self.plot_sentiment_analysis()

        # Sentiment-return correlation
        self.plot_sentiment_return_correlation()

        # Model comparison
        self.plot_model_comparison()

        logger.info("=" * 70)
        logger.info("ALL THESIS VISUALIZATIONS COMPLETE")
        logger.info("=" * 70)


def main():
    viz = ThesisVisualizations()
    viz.generate_all()


if __name__ == "__main__":
    main()
