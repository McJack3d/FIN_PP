"""
SHAP Explainability Analysis (Phase 3)
Computes SHAP values on the Sentiment-Augmented LSTM to identify
which features drive predictions. Generates visualizations for the thesis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Dict, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """Computes and visualizes SHAP values for LSTM models"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.results_path = Path(self.config['paths']['results'])
        self.results_path.mkdir(parents=True, exist_ok=True)

        self.core_set = self.config['tickers']['core_set']
        self.benchmark_set = self.config['tickers']['benchmark_set']
        self.seq_len = self.config['lstm']['sequence_length']

        logger.info("SHAPAnalyzer initialized")

    def compute_shap_values(self, model, X_test: np.ndarray,
                            feature_names: list,
                            n_background: int = 100) -> dict:
        """
        Compute SHAP values using DeepExplainer or KernelExplainer.

        Args:
            model: Trained Keras LSTM model
            X_test: Test sequences (n_samples, seq_len, n_features)
            feature_names: List of feature names
            n_background: Number of background samples for SHAP

        Returns:
            Dict with shap_values and feature_names
        """
        import shap

        n_features = len(feature_names)
        seq_len = X_test.shape[1]

        logger.info(f"Computing SHAP values ({X_test.shape[0]} samples, "
                    f"{n_features} features, seq_len={seq_len})...")

        # Use a subset for background data
        bg_size = min(n_background, X_test.shape[0])
        bg_indices = np.random.choice(X_test.shape[0], bg_size, replace=False)
        background = X_test[bg_indices]

        shap_values = None

        try:
            # Try DeepExplainer first (faster for neural networks)
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(X_test)

            # If output is a list (multi-output), take first
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

        except Exception as e:
            logger.warning(f"DeepExplainer failed ({type(e).__name__}), falling back to KernelExplainer")

            # Flatten sequences for KernelExplainer
            def model_predict(x):
                x_reshaped = x.reshape(-1, seq_len, n_features)
                return model.predict(x_reshaped, verbose=0).flatten()

            X_flat = X_test.reshape(X_test.shape[0], -1)
            bg_flat = background.reshape(background.shape[0], -1)

            explainer = shap.KernelExplainer(model_predict, bg_flat)
            shap_values = explainer.shap_values(X_flat, nsamples=200)

        # Reshape and average over the sequence dimension
        if len(shap_values.shape) == 3:
            # Already (n_samples, seq_len, n_features) from DeepExplainer
            shap_values_avg = np.mean(shap_values, axis=1)
        elif len(shap_values.shape) == 2 and shap_values.shape[1] == seq_len * n_features:
            # Flattened from KernelExplainer: (n_samples, seq_len * n_features)
            # Reshape back to 3D and average over timesteps
            logger.info(f"Reshaping flat SHAP ({shap_values.shape[1]}) -> "
                        f"({seq_len}, {n_features}) and averaging over timesteps")
            shap_3d = shap_values.reshape(shap_values.shape[0], seq_len, n_features)
            shap_values_avg = np.mean(shap_3d, axis=1)
        elif len(shap_values.shape) == 2 and shap_values.shape[1] == n_features:
            # Already aggregated (unlikely but safe)
            shap_values_avg = shap_values
        else:
            logger.warning(f"Unexpected SHAP shape {shap_values.shape} for "
                           f"{n_features} features, seq_len={seq_len}. Attempting reshape.")
            try:
                shap_3d = shap_values.reshape(shap_values.shape[0], seq_len, n_features)
                shap_values_avg = np.mean(shap_3d, axis=1)
            except ValueError:
                logger.error(f"Cannot reshape SHAP values. Skipping.")
                return {}

        logger.info(f"SHAP values computed: shape {shap_values_avg.shape} "
                    f"(matches {n_features} features: {shap_values_avg.shape[1] == n_features})")

        return {
            'shap_values': shap_values_avg,
            'shap_values_raw': shap_values,
            'feature_names': feature_names,
        }

    def plot_summary(self, shap_result: dict, ticker: str, target: str):
        """Generate SHAP summary bar plot showing mean |SHAP| per feature"""
        shap_values = shap_result['shap_values']
        feature_names = shap_result['feature_names']

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        # Handle shape mismatch (flatten if needed)
        if len(mean_abs_shap) != len(feature_names):
            logger.warning(f"SHAP shape mismatch: {len(mean_abs_shap)} vs {len(feature_names)} features")
            return

        # Sort by importance
        sorted_idx = np.argsort(mean_abs_shap)[::-1]

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = range(len(feature_names))

        # Color sentiment features differently
        colors = []
        for idx in sorted_idx:
            name = feature_names[idx]
            if 'Sentiment' in name:
                colors.append('#e74c3c')  # Red for sentiment
            else:
                colors.append('#3498db')  # Blue for price/technical

        ax.barh(y_pos, mean_abs_shap[sorted_idx], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP value|', fontsize=12)
        ax.set_title(f'Feature Importance - {ticker} ({target})', fontsize=14, fontweight='bold')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', label='Sentiment Features'),
            Patch(facecolor='#3498db', label='Price/Technical Features'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        fig_path = self.results_path / f'shap_summary_{ticker}_{target}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP summary plot: {fig_path}")

    def plot_feature_importance_comparison(self,
                                           all_shap_results: Dict,
                                           target: str):
        """
        Compare feature importance across core set vs benchmark set.
        Shows whether sentiment features rank higher for small-caps.
        """
        core_importance = {}
        bench_importance = {}

        for ticker, shap_result in all_shap_results.items():
            if not shap_result:
                continue

            mean_abs = np.mean(np.abs(shap_result['shap_values']), axis=0)
            feature_names = shap_result['feature_names']

            if len(mean_abs) != len(feature_names):
                continue

            importance_dict = dict(zip(feature_names, mean_abs))

            if ticker in self.core_set:
                for f, v in importance_dict.items():
                    core_importance.setdefault(f, []).append(v)
            else:
                for f, v in importance_dict.items():
                    bench_importance.setdefault(f, []).append(v)

        if not core_importance or not bench_importance:
            logger.warning("Insufficient data for comparison plot")
            return

        # Average importance per feature across tickers
        features = sorted(set(core_importance.keys()) & set(bench_importance.keys()))
        core_means = [np.mean(core_importance.get(f, [0])) for f in features]
        bench_means = [np.mean(bench_importance.get(f, [0])) for f in features]

        # Sort by core importance
        sorted_idx = np.argsort(core_means)[::-1]

        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(features))
        width = 0.35

        ax.barh(x - width/2, [core_means[i] for i in sorted_idx],
                width, label='Core Set (Small-Cap)', color='#e74c3c', alpha=0.8)
        ax.barh(x + width/2, [bench_means[i] for i in sorted_idx],
                width, label='Benchmark Set (Large-Cap)', color='#3498db', alpha=0.8)

        ax.set_yticks(x)
        ax.set_yticklabels([features[i] for i in sorted_idx])
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP value|', fontsize=12)
        ax.set_title(f'Feature Importance: Core vs Benchmark ({target})',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)

        plt.tight_layout()
        fig_path = self.results_path / f'shap_comparison_{target}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved comparison plot: {fig_path}")

    def run_shap_analysis(self, experiment_results: Dict) -> Dict:
        """
        Run SHAP analysis on all augmented models from the experiment.

        Args:
            experiment_results: Dict from LSTMForecaster.run_full_experiment()

        Returns:
            Dict of SHAP results per target per ticker
        """
        logger.info("=" * 70)
        logger.info("PHASE 3: SHAP EXPLAINABILITY ANALYSIS")
        logger.info("=" * 70)

        all_shap = {}

        for target_col, target_results in experiment_results.items():
            if not target_results:
                continue

            logger.info(f"\n>>> SHAP analysis for target: {target_col} <<<")
            target_shap = {}

            for ticker, data in target_results.items():
                augmented = data.get('augmented', {})
                if not augmented or 'model' not in augmented:
                    continue

                model = augmented['model']
                features = augmented['features_used']

                # Reconstruct test sequences from the training run
                # We need the processed data to build sequences
                try:
                    processed_path = Path(self.config['paths']['processed'])
                    df = pd.read_csv(processed_path / 'merged_dataset.csv')
                    df['Date'] = pd.to_datetime(df['Date'])

                    df_ticker = df[df['Ticker'] == ticker].sort_values('Date')
                    available_features = [f for f in features if f in df_ticker.columns]

                    df_clean = df_ticker.dropna(subset=available_features + [target_col])

                    # Use test portion
                    test_size = self.config['model']['test_size']
                    split_idx = int(len(df_clean) * (1 - test_size))
                    test_data = df_clean.iloc[split_idx:]

                    X_test = test_data[available_features].values
                    seq_len = self.config['lstm']['sequence_length']

                    # Create sequences
                    X_test_seq = []
                    for i in range(seq_len, len(X_test)):
                        X_test_seq.append(X_test[i - seq_len:i])
                    X_test_seq = np.array(X_test_seq)

                    if len(X_test_seq) < 10:
                        logger.warning(f"Too few test sequences for {ticker} SHAP")
                        continue

                    shap_result = self.compute_shap_values(
                        model, X_test_seq, available_features
                    )

                    self.plot_summary(shap_result, ticker, target_col)
                    target_shap[ticker] = shap_result

                except Exception as e:
                    logger.error(f"SHAP analysis failed for {ticker}: {e}")
                    continue

            if target_shap:
                self.plot_feature_importance_comparison(target_shap, target_col)

            all_shap[target_col] = target_shap

        # Save SHAP importance rankings
        self._save_importance_rankings(all_shap)

        logger.info("\n" + "=" * 70)
        logger.info("SHAP ANALYSIS COMPLETE")
        logger.info("=" * 70)

        return all_shap

    def _save_importance_rankings(self, all_shap: Dict):
        """Save feature importance rankings to CSV"""
        rows = []
        for target, target_shap in all_shap.items():
            for ticker, shap_result in target_shap.items():
                if not shap_result:
                    continue

                mean_abs = np.mean(np.abs(shap_result['shap_values']), axis=0)
                features = shap_result['feature_names']

                if len(mean_abs) != len(features):
                    continue

                for f, v in zip(features, mean_abs):
                    group = 'core_set' if ticker in self.core_set else 'benchmark_set'
                    rows.append({
                        'Target': target,
                        'Ticker': ticker,
                        'Group': group,
                        'Feature': f,
                        'Mean_Abs_SHAP': float(v),
                    })

        if rows:
            df = pd.DataFrame(rows)
            path = self.results_path / 'shap_importance_rankings.csv'
            df.to_csv(path, index=False)
            logger.info(f"Saved importance rankings to {path}")


def main():
    analyzer = SHAPAnalyzer()
    logger.info("SHAP Analyzer ready. Run via run_all.py for full pipeline.")


if __name__ == "__main__":
    main()
