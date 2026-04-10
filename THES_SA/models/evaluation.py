"""
Model Evaluation (Phase 3)
Computes MAE, Directional Accuracy, Diebold-Mariano test.
Tests H1 (sentiment improves prediction) and H2 (Small-Cap Sentiment Premium).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Dict, Optional
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates and compares Baseline vs Sentiment-Augmented LSTM models"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.results_path = Path(self.config['paths']['results'])
        self.results_path.mkdir(parents=True, exist_ok=True)

        self.core_set = self.config['tickers']['core_set']
        self.benchmark_set = self.config['tickers']['benchmark_set']

        logger.info("ModelEvaluator initialized")

    @staticmethod
    def compute_mae(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return np.mean(np.abs(y_actual - y_pred))

    @staticmethod
    def compute_rmse(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((y_actual - y_pred) ** 2))

    @staticmethod
    def compute_directional_accuracy(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Directional Accuracy: percentage of correct up/down predictions.
        Compares sign of actual vs predicted return.
        """
        correct = np.sign(y_actual) == np.sign(y_pred)
        return np.mean(correct) * 100

    @staticmethod
    def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray,
                             h: int = 1, power: int = 2) -> dict:
        """
        Diebold-Mariano test for equal predictive accuracy.

        H0: Both models have equal forecast accuracy.
        H1: Models have different forecast accuracy.

        Args:
            e1: Forecast errors from model 1 (baseline)
            e2: Forecast errors from model 2 (augmented)
            h: Forecast horizon
            power: Loss function power (1=MAE, 2=MSE)

        Returns:
            Dict with test statistic, p-value, and interpretation
        """
        d = np.abs(e1) ** power - np.abs(e2) ** power
        n = len(d)

        if n < 10:
            return {
                'dm_statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'interpretation': 'Insufficient data for DM test'
            }

        d_mean = np.mean(d)
        d_var = np.var(d, ddof=1)

        # Newey-West style variance adjustment for autocorrelation
        if h > 1:
            for k in range(1, h):
                gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
                d_var += 2 * gamma_k / n

        if d_var <= 0:
            return {
                'dm_statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'interpretation': 'Zero variance in loss differential'
            }

        dm_stat = d_mean / np.sqrt(d_var / n)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        significant = p_value < 0.05

        if significant:
            if dm_stat > 0:
                interp = "Augmented model significantly outperforms baseline (p<0.05)"
            else:
                interp = "Baseline model significantly outperforms augmented (p<0.05)"
        else:
            interp = "No significant difference between models (p>=0.05)"

        return {
            'dm_statistic': float(dm_stat),
            'p_value': float(p_value),
            'significant': significant,
            'interpretation': interp
        }

    def evaluate_experiment(self, results: Dict, target_col: str) -> pd.DataFrame:
        """
        Evaluate all models from an experiment.

        Args:
            results: Dict from LSTMForecaster.run_experiment()
            target_col: Target column name

        Returns:
            DataFrame with evaluation metrics per ticker
        """
        logger.info(f"\nEvaluating models for target: {target_col}")
        logger.info("=" * 70)

        eval_rows = []

        for ticker, data in results.items():
            baseline = data.get('baseline', {})
            augmented = data.get('augmented', {})

            if not baseline or not augmented:
                logger.warning(f"Skipping {ticker}: missing model results")
                continue

            y_actual = baseline['y_actual']
            y_pred_base = baseline['y_pred']
            y_pred_aug = augmented['y_pred']

            # Ensure same length (use shorter)
            min_len = min(len(y_actual), len(y_pred_base), len(y_pred_aug))
            y_actual = y_actual[:min_len]
            y_pred_base = y_pred_base[:min_len]
            y_pred_aug = y_pred_aug[:min_len]

            # Metrics - Baseline
            mae_base = self.compute_mae(y_actual, y_pred_base)
            rmse_base = self.compute_rmse(y_actual, y_pred_base)
            da_base = self.compute_directional_accuracy(y_actual, y_pred_base)

            # Metrics - Augmented
            mae_aug = self.compute_mae(y_actual, y_pred_aug)
            rmse_aug = self.compute_rmse(y_actual, y_pred_aug)
            da_aug = self.compute_directional_accuracy(y_actual, y_pred_aug)

            # Diebold-Mariano test
            errors_base = y_actual - y_pred_base
            errors_aug = y_actual - y_pred_aug
            horizon = 5 if '5d' in target_col else 1
            dm_result = self.diebold_mariano_test(errors_base, errors_aug, h=horizon)

            # MAE improvement
            mae_improvement = mae_base - mae_aug
            mae_improvement_pct = (mae_improvement / mae_base * 100) if mae_base > 0 else 0

            row = {
                'Ticker': ticker,
                'Group': data['group'],
                'Target': target_col,
                'MAE_Baseline': mae_base,
                'MAE_Augmented': mae_aug,
                'MAE_Improvement': mae_improvement,
                'MAE_Improvement_Pct': mae_improvement_pct,
                'RMSE_Baseline': rmse_base,
                'RMSE_Augmented': rmse_aug,
                'DA_Baseline': da_base,
                'DA_Augmented': da_aug,
                'DM_Statistic': dm_result['dm_statistic'],
                'DM_PValue': dm_result['p_value'],
                'DM_Significant': dm_result['significant'],
                'DM_Interpretation': dm_result['interpretation'],
                'N_Test': min_len,
            }
            eval_rows.append(row)

            logger.info(f"\n{ticker} ({data['group']}):")
            logger.info(f"  Baseline  - MAE: {mae_base:.6f}, DA: {da_base:.1f}%")
            logger.info(f"  Augmented - MAE: {mae_aug:.6f}, DA: {da_aug:.1f}%")
            logger.info(f"  MAE Improvement: {mae_improvement_pct:+.2f}%")
            logger.info(f"  DM test: {dm_result['interpretation']}")

        df_eval = pd.DataFrame(eval_rows)
        return df_eval

    def test_h1(self, df_eval: pd.DataFrame) -> dict:
        """
        Test H1: Does sentiment improve prediction for core set?
        H1 passes if the average MAE improvement is positive and
        the majority of core set tickers show improvement.
        """
        logger.info("\n" + "=" * 70)
        logger.info("HYPOTHESIS H1: Sentiment improves short-term return prediction")
        logger.info("=" * 70)

        core = df_eval[df_eval['Group'] == 'core_set']

        if core.empty:
            return {'supported': False, 'reason': 'No core set results'}

        avg_improvement = core['MAE_Improvement_Pct'].mean()
        tickers_improved = (core['MAE_Improvement'] > 0).sum()
        tickers_significant = core['DM_Significant'].sum()
        total = len(core)

        supported = avg_improvement > 0 and tickers_improved > total / 2

        result = {
            'supported': bool(supported),
            'avg_mae_improvement_pct': float(avg_improvement),
            'tickers_improved': int(tickers_improved),
            'tickers_significant': int(tickers_significant),
            'total_tickers': int(total),
            'avg_da_baseline': float(core['DA_Baseline'].mean()),
            'avg_da_augmented': float(core['DA_Augmented'].mean()),
        }

        status = "SUPPORTED" if supported else "NOT SUPPORTED"
        logger.info(f"H1 Result: {status}")
        logger.info(f"  Avg MAE improvement: {avg_improvement:+.2f}%")
        logger.info(f"  Tickers improved: {tickers_improved}/{total}")
        logger.info(f"  Statistically significant: {tickers_significant}/{total}")

        return result

    def test_h2(self, df_eval: pd.DataFrame) -> dict:
        """
        Test H2 (Small-Cap Sentiment Premium):
        Is the MAE improvement from sentiment significantly larger for
        core set (small-caps) than benchmark set (large-caps)?
        """
        logger.info("\n" + "=" * 70)
        logger.info("HYPOTHESIS H2: Small-Cap Sentiment Premium")
        logger.info("=" * 70)

        core = df_eval[df_eval['Group'] == 'core_set']['MAE_Improvement_Pct']
        bench = df_eval[df_eval['Group'] == 'benchmark_set']['MAE_Improvement_Pct']

        if core.empty or bench.empty:
            return {'supported': False, 'reason': 'Missing group data'}

        core_mean = core.mean()
        bench_mean = bench.mean()
        premium = core_mean - bench_mean

        # Two-sample t-test (one-sided: core > bench)
        if len(core) >= 2 and len(bench) >= 2:
            t_stat, p_value_two = stats.ttest_ind(core, bench, equal_var=False)
            # One-sided p-value (core > bench)
            p_value = p_value_two / 2 if t_stat > 0 else 1 - p_value_two / 2
        else:
            t_stat, p_value = np.nan, np.nan

        supported = premium > 0 and (p_value < 0.10 if not np.isnan(p_value) else False)

        result = {
            'supported': bool(supported),
            'core_avg_improvement_pct': float(core_mean),
            'benchmark_avg_improvement_pct': float(bench_mean),
            'sentiment_premium_pct': float(premium),
            't_statistic': float(t_stat) if not np.isnan(t_stat) else None,
            'p_value': float(p_value) if not np.isnan(p_value) else None,
        }

        status = "SUPPORTED" if supported else "NOT SUPPORTED"
        logger.info(f"H2 Result: {status}")
        logger.info(f"  Core Set avg improvement: {core_mean:+.2f}%")
        logger.info(f"  Benchmark avg improvement: {bench_mean:+.2f}%")
        logger.info(f"  Sentiment Premium: {premium:+.2f}%")
        if not np.isnan(p_value):
            logger.info(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")

        return result

    def run_full_evaluation(self, results: Dict) -> dict:
        """
        Run evaluation for all targets and test both hypotheses.

        Args:
            results: Dict from LSTMForecaster.run_full_experiment()

        Returns:
            Complete evaluation report
        """
        logger.info("=" * 70)
        logger.info("PHASE 3: MODEL EVALUATION & HYPOTHESIS TESTING")
        logger.info("=" * 70)

        full_report = {}

        for target_col, target_results in results.items():
            if not target_results:
                continue

            logger.info(f"\n>>> Evaluating target: {target_col} <<<")

            df_eval = self.evaluate_experiment(target_results, target_col)

            if df_eval.empty:
                continue

            h1 = self.test_h1(df_eval)
            h2 = self.test_h2(df_eval)

            full_report[target_col] = {
                'evaluation_metrics': df_eval.to_dict('records'),
                'h1_result': h1,
                'h2_result': h2,
            }

            # Save evaluation table
            eval_path = self.results_path / f'evaluation_{target_col}.csv'
            df_eval.to_csv(eval_path, index=False)
            logger.info(f"Saved evaluation to {eval_path}")

        # Save full report
        self._save_report(full_report)

        return full_report

    def _save_report(self, report: dict):
        """Save evaluation report as YAML"""
        # Convert numpy types for YAML serialization
        def clean_for_yaml(obj):
            if isinstance(obj, dict):
                return {k: clean_for_yaml(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_yaml(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        report_clean = clean_for_yaml(report)

        report_path = self.results_path / 'evaluation_report.yaml'
        with open(report_path, 'w') as f:
            yaml.dump(report_clean, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved evaluation report to {report_path}")


def main():
    """Standalone evaluation from saved predictions"""
    evaluator = ModelEvaluator()

    results_path = Path(evaluator.config['paths']['results'])

    for target in evaluator.config['model']['target_columns']:
        pred_file = results_path / f'predictions_{target}.csv'
        if pred_file.exists():
            logger.info(f"Loading predictions for {target}...")
            df = pd.read_csv(pred_file)
            # Reconstruct results dict for evaluation
            # (This is for standalone re-evaluation from saved predictions)
            logger.info(f"Found {len(df)} prediction rows for {target}")
        else:
            logger.warning(f"No predictions found for {target}")


if __name__ == "__main__":
    main()
