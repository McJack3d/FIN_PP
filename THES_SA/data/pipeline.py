"""
Main Data Engineering Pipeline (v2.0)
Orchestrates data collection, preprocessing, and feasibility audit.
Daily frequency only per advisor feedback.
"""

import sys
from pathlib import Path
import yaml
import logging
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from quantitative_collector import QuantitativeDataCollector
from textual_collector import TextualDataCollector
from preprocessing import FinancialDataPreprocessor, TextualDataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataEngineeringPipeline:
    """
    Main pipeline for synchronizing quantitative and textual data streams (v2.0)
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the complete data engineering pipeline"""
        self.config_path = config_path

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize collectors
        self.quant_collector = QuantitativeDataCollector(config_path)
        self.text_collector = TextualDataCollector(config_path)

        # Initialize preprocessors
        self.fin_preprocessor = FinancialDataPreprocessor(config_path)
        self.text_preprocessor = TextualDataPreprocessor(config_path)

        logger.info("Data Engineering Pipeline initialized (v2.0 - Daily only)")
        logger.info(f"Core Set: {self.config['tickers']['core_set']}")
        logger.info(f"Benchmark Set: {self.config['tickers']['benchmark_set']}")

    def run_feasibility_audit(self):
        """
        Run Phase 0: Data Feasibility Audit
        Checks FNSPID coverage per ticker before committing to full pipeline.
        """
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 0: DATA FEASIBILITY AUDIT")
        logger.info("=" * 70)

        try:
            from feasibility_audit import FeasibilityAuditor
            auditor = FeasibilityAuditor(self.config_path)
            report = auditor.run_audit()
            return report
        except ImportError:
            logger.error("feasibility_audit module not found")
            return None
        except Exception as e:
            logger.error(f"Feasibility audit failed: {e}")
            return None

    def run_data_collection(self,
                           collect_quantitative: bool = True,
                           collect_textual: bool = True,
                           use_newsapi: bool = False,
                           use_scraping: bool = True):
        """
        Run Phase 1: Data Collection (daily frequency only)

        Args:
            collect_quantitative: Whether to collect financial data
            collect_textual: Whether to collect news data
            use_newsapi: Whether to use NewsAPI
            use_scraping: Whether to use web scraping
        """
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1: DATA COLLECTION")
        logger.info("=" * 70)

        start_time = datetime.now()

        # Quantitative stream (daily only)
        if collect_quantitative:
            logger.info("\n>>> QUANTITATIVE STREAM (Daily) <<<")
            try:
                self.quant_collector.collect_all()
                logger.info("Quantitative data collection completed")
            except Exception as e:
                logger.error(f"Quantitative collection failed: {e}")

        # Textual stream
        if collect_textual:
            logger.info("\n>>> TEXTUAL STREAM <<<")
            try:
                self.text_collector.collect_all(
                    use_newsapi=use_newsapi,
                    use_scraping=use_scraping
                )
                logger.info("Textual data collection completed")
            except Exception as e:
                logger.error(f"Textual collection failed: {e}")

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\nData collection completed in {elapsed:.1f} seconds")

    def run_preprocessing(self,
                         process_financial: bool = True,
                         process_textual: bool = True):
        """
        Run Phase 1b: Data Preprocessing

        Args:
            process_financial: Whether to preprocess financial data
            process_textual: Whether to preprocess textual data
        """
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1b: DATA PREPROCESSING")
        logger.info("=" * 70)

        start_time = datetime.now()

        # Financial preprocessing (daily only)
        if process_financial:
            logger.info("\n>>> FINANCIAL DATA PREPROCESSING <<<")
            try:
                self.fin_preprocessor.process_pipeline(
                    'daily_ohlcv.csv',
                    'daily_ohlcv_processed.csv'
                )
                logger.info("Financial preprocessing completed")
            except Exception as e:
                logger.error(f"Financial preprocessing failed: {e}")

        # Textual preprocessing
        if process_textual:
            logger.info("\n>>> TEXTUAL DATA PREPROCESSING <<<")
            try:
                self.text_preprocessor.process_pipeline(
                    'all_news_combined.csv',
                    'news_processed.csv'
                )
                logger.info("Textual preprocessing completed")
            except Exception as e:
                logger.error(f"Textual preprocessing failed: {e}")

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\nPreprocessing completed in {elapsed:.1f} seconds")

    def run_full_pipeline(self,
                         use_newsapi: bool = False,
                         use_scraping: bool = True,
                         skip_audit: bool = False):
        """
        Run complete data engineering pipeline (Phases 0 + 1)

        Args:
            use_newsapi: Whether to use NewsAPI
            use_scraping: Whether to use web scraping
            skip_audit: Whether to skip Phase 0 feasibility audit
        """
        logger.info("\n" + "=" * 70)
        logger.info("COMPLETE DATA ENGINEERING PIPELINE (v2.0)")
        logger.info("=" * 70)
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        pipeline_start = datetime.now()

        # Phase 0: Feasibility Audit
        if not skip_audit:
            self.run_feasibility_audit()

        # Phase 1: Data Collection (daily only)
        self.run_data_collection(
            collect_quantitative=True,
            collect_textual=True,
            use_newsapi=use_newsapi,
            use_scraping=use_scraping
        )

        # Phase 1b: Data Preprocessing
        self.run_preprocessing(
            process_financial=True,
            process_textual=True
        )

        # Pipeline summary
        total_elapsed = (datetime.now() - pipeline_start).total_seconds()

        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total execution time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.print_data_summary()

        logger.info("\n" + "=" * 70)
        logger.info("DATA ENGINEERING PIPELINE COMPLETE")
        logger.info("=" * 70)

    def print_data_summary(self):
        """Print summary of collected and processed data"""
        import pandas as pd

        logger.info("\n--- Data Summary ---")

        processed_path = Path(self.config['paths']['processed'])

        files_to_check = [
            ('daily_ohlcv_processed.csv', 'Daily OHLCV (Processed)'),
            ('news_processed.csv', 'News Articles (Processed)')
        ]

        for filename, description in files_to_check:
            filepath = processed_path / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    logger.info(f"{description}: {len(df)} records")

                    if 'Ticker' in df.columns:
                        tickers = df['Ticker'].unique()
                        logger.info(f"  - {len(tickers)} tickers: {', '.join(tickers)}")

                    if 'Date' in df.columns or 'date' in df.columns:
                        date_col = 'Date' if 'Date' in df.columns else 'date'
                        df[date_col] = pd.to_datetime(df[date_col])
                        logger.info(f"  - Date range: {df[date_col].min()} to {df[date_col].max()}")

                except Exception as e:
                    logger.warning(f"Could not read {filename}: {e}")
            else:
                logger.warning(f"{description}: File not found")


def main():
    """Main execution with command line arguments"""
    parser = argparse.ArgumentParser(
        description='Data Engineering Pipeline for Financial Analysis (v2.0)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'collect', 'preprocess', 'audit'],
        default='full',
        help='Pipeline mode: full (all phases), collect (data collection only), '
             'preprocess (preprocessing only), audit (Phase 0 feasibility audit only)'
    )

    parser.add_argument(
        '--use-newsapi',
        action='store_true',
        help='Use NewsAPI for news collection (requires API key)'
    )

    parser.add_argument(
        '--no-scraping',
        action='store_true',
        help='Disable web scraping'
    )

    parser.add_argument(
        '--skip-audit',
        action='store_true',
        help='Skip Phase 0 feasibility audit'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = DataEngineeringPipeline(config_path=args.config)

    # Run based on mode
    if args.mode == 'full':
        pipeline.run_full_pipeline(
            use_newsapi=args.use_newsapi,
            use_scraping=not args.no_scraping,
            skip_audit=args.skip_audit
        )
    elif args.mode == 'collect':
        pipeline.run_data_collection(
            collect_quantitative=True,
            collect_textual=True,
            use_newsapi=args.use_newsapi,
            use_scraping=not args.no_scraping
        )
    elif args.mode == 'preprocess':
        pipeline.run_preprocessing(
            process_financial=True,
            process_textual=True
        )
    elif args.mode == 'audit':
        pipeline.run_feasibility_audit()


if __name__ == "__main__":
    main()
