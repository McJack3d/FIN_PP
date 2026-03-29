"""
THES_SA Kaggle Kernel Entry Point
Sentiment-Driven Nuclear Equity Forecasting (v2.0)

Run on Kaggle with:
  - GPU enabled (P100 or T4)
  - Internet enabled (for yfinance downloads)
  - Dataset: elsabetyemane/financial-news-and-stock-price-integration-dataset (FNSPID)
  - Dataset: alexandrebredillot/thes-sa-code (project source code)

Usage (Kaggle):
  Attached as code_file in kernel-metadata.json, executed via `kaggle kernels push`.

Usage (local, for testing):
  python kaggle_notebook.py [--phase N] [--start-from N]
"""

import os
import sys
import shutil
import subprocess
import logging
import argparse
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('THES_SA_KAGGLE')

# ============================================================
# ENVIRONMENT DETECTION
# ============================================================
IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
WORKING_DIR = Path('/kaggle/working') if IS_KAGGLE else Path(__file__).parent
FNSPID_INPUT = Path('/kaggle/input/financial-news-and-stock-price-integration-dataset')
CODE_INPUT = Path('/kaggle/input/thes-sa-code')


def print_environment():
    """Print runtime environment details."""
    logger.info("=" * 70)
    logger.info("THES_SA - RUNTIME ENVIRONMENT")
    logger.info("=" * 70)
    logger.info(f"Running on Kaggle: {IS_KAGGLE}")
    logger.info(f"Working directory: {WORKING_DIR}")
    logger.info(f"Python: {sys.version}")

    # GPU check - PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"PyTorch GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
            logger.info(f"  VRAM: {vram / 1e9:.1f} GB")
        else:
            logger.info("PyTorch GPU: not available")
    except ImportError:
        logger.info("PyTorch: not installed")

    # GPU check - TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"TensorFlow GPUs: {len(gpus)}")
        else:
            logger.info("TensorFlow GPU: not available")
    except ImportError:
        logger.info("TensorFlow: not installed")

    # FNSPID dataset
    if FNSPID_INPUT.exists():
        csv_count = len(list(FNSPID_INPUT.glob('**/*.csv')))
        logger.info(f"FNSPID dataset: {FNSPID_INPUT} ({csv_count} CSV files)")
    else:
        logger.info("FNSPID dataset: not found at input path")

    # Code dataset
    if CODE_INPUT.exists():
        logger.info(f"Code dataset: {CODE_INPUT}")
    else:
        logger.info("Code dataset: not found (using local files)")

    logger.info("=" * 70)


def install_dependencies():
    """Install required packages not available by default on Kaggle."""
    packages = [
        'pandas-ta',
        'shap',
    ]
    for pkg in packages:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            logger.info(f"Installing {pkg}...")
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '-q', pkg],
                stdout=subprocess.DEVNULL
            )


def setup_code():
    """
    Copy project source code to the working directory.
    On Kaggle, code is read from /kaggle/input/thes-sa-code/.
    Locally, code is already in the same directory.
    """
    if not IS_KAGGLE:
        # Local: just ensure project root is on path
        sys.path.insert(0, str(WORKING_DIR))
        sys.path.insert(0, str(WORKING_DIR / 'data'))
        sys.path.insert(0, str(WORKING_DIR / 'sentiment'))
        sys.path.insert(0, str(WORKING_DIR / 'models'))
        return

    logger.info("Setting up code in Kaggle working directory...")

    # Copy source modules from code dataset
    src_root = CODE_INPUT
    if not src_root.exists():
        # Fallback: maybe the code is nested one level
        candidates = list(Path('/kaggle/input').glob('*/THES_SA'))
        if candidates:
            src_root = candidates[0].parent
        else:
            logger.error("Code dataset not found. Upload THES_SA as a Kaggle dataset first.")
            logger.info("Run: cd THES_SA && kaggle datasets create -p . -r zip")
            sys.exit(1)

    for subdir in ['data', 'sentiment', 'models']:
        src = src_root / subdir
        dst = WORKING_DIR / subdir
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            logger.info(f"  Copied {subdir}/ -> {dst}")
        else:
            # Try nested THES_SA/subdir
            src_nested = src_root / 'THES_SA' / subdir
            if src_nested.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src_nested, dst)
                logger.info(f"  Copied THES_SA/{subdir}/ -> {dst}")

    # Also copy standalone files
    for filename in ['run_all.py', 'requirements.txt']:
        for src_file in [src_root / filename, src_root / 'THES_SA' / filename]:
            if src_file.exists():
                shutil.copy2(src_file, WORKING_DIR / filename)
                break

    # Add to path
    sys.path.insert(0, str(WORKING_DIR))
    sys.path.insert(0, str(WORKING_DIR / 'data'))
    sys.path.insert(0, str(WORKING_DIR / 'sentiment'))
    sys.path.insert(0, str(WORKING_DIR / 'models'))


def generate_config():
    """
    Generate a Kaggle-aware config.yaml in the working directory.
    Adjusts all paths for /kaggle/working/ and points FNSPID to the input dataset.
    """
    import yaml

    config_src = WORKING_DIR / 'config.yaml'

    # Try loading existing config
    if config_src.exists():
        with open(config_src) as f:
            config = yaml.safe_load(f)
    else:
        # Try from code dataset
        for candidate in [CODE_INPUT / 'config.yaml', CODE_INPUT / 'THES_SA' / 'config.yaml']:
            if candidate.exists():
                with open(candidate) as f:
                    config = yaml.safe_load(f)
                break
        else:
            logger.error("config.yaml not found anywhere")
            sys.exit(1)

    if IS_KAGGLE:
        # Override paths for Kaggle
        config['paths']['root'] = str(WORKING_DIR)
        config['paths']['data_dir'] = 'data'
        config['paths']['raw'] = 'data/raw'
        config['paths']['processed'] = 'data/processed'
        config['paths']['news'] = 'data/news'
        config['paths']['models'] = 'models'
        config['paths']['results'] = 'results'
        config['paths']['cache'] = 'cache'

        # Kaggle-specific settings
        config['kaggle']['fnspid_input_path'] = str(FNSPID_INPUT)
        config['kaggle']['working_dir'] = str(WORKING_DIR)

        # HuggingFace cache in working dir
        config['huggingface']['cache_dir'] = str(WORKING_DIR / 'hf_cache')

    config_dst = WORKING_DIR / 'config.yaml'
    with open(config_dst, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Config written to {config_dst}")

    # Create required directories
    for dir_key in ['raw', 'processed', 'news', 'models', 'results', 'cache']:
        (WORKING_DIR / config['paths'][dir_key]).mkdir(parents=True, exist_ok=True)

    return config_dst


# ============================================================
# PIPELINE PHASES
# ============================================================

def run_phase_0(config_path):
    """Phase 0: Data Feasibility Audit"""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 0: DATA FEASIBILITY AUDIT")
    logger.info("=" * 70)

    from data.feasibility_audit import FeasibilityAuditor
    auditor = FeasibilityAuditor(str(config_path))
    report = auditor.run_audit()

    if report and report.get('failing_tickers', 0) > 0:
        logger.warning(
            f"{report['failing_tickers']} ticker(s) below article threshold. "
            f"Check results/feasibility_audit.yaml for details."
        )
    return report


def run_phase_1(config_path):
    """Phase 1: Data Collection + Preprocessing"""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: DATA COLLECTION + PREPROCESSING")
    logger.info("=" * 70)

    from data.pipeline import DataEngineeringPipeline
    pipeline = DataEngineeringPipeline(str(config_path))
    pipeline.run_full_pipeline(skip_audit=True)

    try:
        from data.generate_summary import PipelineSummary
        summary = PipelineSummary(config_path=str(config_path))
        summary.generate_full_report()
    except Exception as e:
        logger.warning(f"Summary generation failed (non-critical): {e}")


def run_phase_2(config_path):
    """Phase 2: Sentiment Scoring + Feature Engineering"""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: SENTIMENT ANALYSIS")
    logger.info("=" * 70)

    from sentiment.scorer import SentimentScorer
    scorer = SentimentScorer(str(config_path))
    df_scored = scorer.run_scoring_pipeline()

    if df_scored.empty:
        logger.error("Sentiment scoring failed.")
        return False

    from sentiment.features import SentimentFeatureEngineer
    engineer = SentimentFeatureEngineer(str(config_path))
    df_merged = engineer.run_feature_pipeline()

    if df_merged.empty:
        logger.error("Sentiment feature engineering failed.")
        return False

    return True


def run_phase_3(config_path):
    """Phase 3: LSTM Modeling + Evaluation + SHAP"""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: LSTM MODELING + EVALUATION")
    logger.info("=" * 70)

    from models.lstm_model import LSTMForecaster
    forecaster = LSTMForecaster(str(config_path))
    experiment_results = forecaster.run_full_experiment()

    if not experiment_results:
        logger.error("LSTM training failed.")
        return None

    from models.evaluation import ModelEvaluator
    evaluator = ModelEvaluator(str(config_path))
    eval_report = evaluator.run_full_evaluation(experiment_results)

    try:
        from models.explainability import SHAPAnalyzer
        analyzer = SHAPAnalyzer(str(config_path))
        analyzer.run_shap_analysis(experiment_results)
    except Exception as e:
        logger.warning(f"SHAP analysis failed (non-critical): {e}")

    return eval_report


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='THES_SA Kaggle Kernel (v2.0)')
    parser.add_argument('--phase', type=int, choices=[0, 1, 2, 3],
                        help='Run only a specific phase')
    parser.add_argument('--start-from', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Start from a specific phase')
    args = parser.parse_args()

    start_time = datetime.now()

    # Setup
    print_environment()
    install_dependencies()
    setup_code()
    config_path = generate_config()

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
        name, func = phases[args.phase]
        logger.info(f"\nRunning Phase {args.phase}: {name}")
        func(config_path)
    else:
        for phase_num in range(args.start_from, 4):
            name, func = phases[phase_num]
            logger.info(f"\n>>> Phase {phase_num}: {name}")
            result = func(config_path)

            if phase_num == 2 and result is False:
                logger.error("Phase 2 failed. Stopping pipeline.")
                break

    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    logger.info("\nOutput locations:")
    logger.info("  Processed data:  data/processed/")
    logger.info("  Trained models:  models/")
    logger.info("  Results & plots: results/")
    logger.info("  Evaluation:      results/evaluation_report.yaml")

    if IS_KAGGLE:
        logger.info("\nAll outputs are saved in /kaggle/working/ and will be")
        logger.info("available as kernel output after execution completes.")


if __name__ == "__main__":
    main()
