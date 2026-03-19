"""
THES_SA Master Pipeline Orchestrator (v2.0)
Runs all phases sequentially:
  Phase 0: Feasibility Audit
  Phase 1: Data Collection + Preprocessing
  Phase 2: Sentiment Scoring + Feature Engineering
  Phase 3: LSTM Modeling + Evaluation + SHAP

Usage:
  python run_all.py                    # Run all phases
  python run_all.py --start-from 2     # Resume from Phase 2
  python run_all.py --phase 0          # Run only Phase 0
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'data'))
sys.path.insert(0, str(PROJECT_ROOT / 'sentiment'))
sys.path.insert(0, str(PROJECT_ROOT / 'models'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('THES_SA')

CONFIG_PATH = str(PROJECT_ROOT / 'config.yaml')


def run_phase_0():
    """Phase 0: Data Feasibility Audit"""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 0: DATA FEASIBILITY AUDIT")
    logger.info("=" * 70)

    from data.feasibility_audit import FeasibilityAuditor
    auditor = FeasibilityAuditor(CONFIG_PATH)
    report = auditor.run_audit()

    if report and report.get('failing_tickers', 0) > 0:
        logger.warning(
            f"\n{report['failing_tickers']} ticker(s) below article threshold. "
            f"Check results/feasibility_audit.yaml for details.\n"
            f"You may want to supplement via scraping or adjust the ticker list "
            f"before proceeding."
        )

    return report


def run_phase_1():
    """Phase 1: Data Collection + Preprocessing"""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: DATA COLLECTION + PREPROCESSING")
    logger.info("=" * 70)

    from data.pipeline import DataEngineeringPipeline
    pipeline = DataEngineeringPipeline(CONFIG_PATH)
    pipeline.run_full_pipeline(skip_audit=True)

    # Generate summary visualizations
    try:
        from data.generate_summary import PipelineSummary
        summary = PipelineSummary(config_path=CONFIG_PATH)
        summary.generate_full_report()
    except Exception as e:
        logger.warning(f"Summary generation failed (non-critical): {e}")


def run_phase_2():
    """Phase 2: Sentiment Scoring + Feature Engineering"""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: SENTIMENT ANALYSIS")
    logger.info("=" * 70)

    # Phase 2a: FinBERT Scoring
    from sentiment.scorer import SentimentScorer
    scorer = SentimentScorer(CONFIG_PATH)
    df_scored = scorer.run_scoring_pipeline()

    if df_scored.empty:
        logger.error("Sentiment scoring failed. Cannot proceed to feature engineering.")
        return False

    # Phase 2b: Feature Engineering + Merge
    from sentiment.features import SentimentFeatureEngineer
    engineer = SentimentFeatureEngineer(CONFIG_PATH)
    df_merged = engineer.run_feature_pipeline()

    if df_merged.empty:
        logger.error("Sentiment feature engineering failed.")
        return False

    return True


def run_phase_3():
    """Phase 3: LSTM Modeling + Evaluation + SHAP"""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: LSTM MODELING + EVALUATION")
    logger.info("=" * 70)

    # Phase 3a: Train LSTM models
    from models.lstm_model import LSTMForecaster
    forecaster = LSTMForecaster(CONFIG_PATH)
    experiment_results = forecaster.run_full_experiment()

    if not experiment_results:
        logger.error("LSTM training failed. No results to evaluate.")
        return

    # Phase 3b: Evaluate and test hypotheses
    from models.evaluation import ModelEvaluator
    evaluator = ModelEvaluator(CONFIG_PATH)
    eval_report = evaluator.run_full_evaluation(experiment_results)

    # Phase 3c: SHAP Explainability
    try:
        from models.explainability import SHAPAnalyzer
        analyzer = SHAPAnalyzer(CONFIG_PATH)
        analyzer.run_shap_analysis(experiment_results)
    except Exception as e:
        logger.warning(f"SHAP analysis failed (non-critical): {e}")
        logger.info("You can re-run SHAP separately after fixing the issue.")

    return eval_report


def main():
    parser = argparse.ArgumentParser(description='THES_SA Full Pipeline (v2.0)')

    parser.add_argument(
        '--phase', type=int, choices=[0, 1, 2, 3],
        help='Run only a specific phase'
    )
    parser.add_argument(
        '--start-from', type=int, choices=[0, 1, 2, 3], default=0,
        help='Start from a specific phase (runs all subsequent phases)'
    )

    args = parser.parse_args()

    start_time = datetime.now()

    logger.info("=" * 70)
    logger.info("THES_SA: Sentiment-Driven Nuclear Equity Forecasting (v2.0)")
    logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    phases = {
        0: ('Feasibility Audit', run_phase_0),
        1: ('Data Collection + Preprocessing', run_phase_1),
        2: ('Sentiment Analysis', run_phase_2),
        3: ('LSTM Modeling + Evaluation', run_phase_3),
    }

    if args.phase is not None:
        # Run single phase
        name, func = phases[args.phase]
        logger.info(f"\nRunning Phase {args.phase}: {name}")
        func()
    else:
        # Run from start_from through all remaining
        for phase_num in range(args.start_from, 4):
            name, func = phases[phase_num]
            logger.info(f"\n>>> Phase {phase_num}: {name}")
            result = func()

            # Check for critical failures
            if phase_num == 2 and result is False:
                logger.error("Phase 2 failed. Stopping pipeline.")
                break

    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    logger.info("\nOutput locations:")
    logger.info("  Processed data:  data/processed/")
    logger.info("  Trained models:  models/")
    logger.info("  Results & plots: results/")
    logger.info("  Evaluation:      results/evaluation_report.yaml")


if __name__ == "__main__":
    main()
