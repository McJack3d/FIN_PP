"""
Phase 0: Data Feasibility Audit
Checks FNSPID news coverage per ticker before committing to full pipeline.
Tickers with <50 articles are flagged for supplementation or removal.
"""

import sys
from pathlib import Path
import yaml
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeasibilityAuditor:
    """Audits FNSPID dataset coverage for each ticker in the study universe"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.core_set = self.config['tickers']['core_set']
        self.benchmark_set = self.config['tickers']['benchmark_set']
        self.all_tickers = self.config['tickers']['all_tickers']
        self.min_articles = self.config['phase0']['min_articles_per_ticker']

        self.results_path = Path(self.config['paths']['results'])
        self.results_path.mkdir(parents=True, exist_ok=True)

    def _compute_dates(self) -> tuple:
        start_date = self.config['dates'].get('start_date')
        end_date = self.config['dates'].get('end_date')
        months_back = self.config['dates']['months_back']

        if start_date and end_date:
            return pd.to_datetime(start_date), pd.to_datetime(end_date)
        else:
            end = datetime.now()
            start = end - timedelta(days=months_back * 30)
            return start, end

    def _load_fnspid_local(self) -> pd.DataFrame:
        """Try to load FNSPID from local path (Kaggle input or manual download)."""
        kaggle_cfg = self.config.get('kaggle', {})
        fnspid_path = Path(kaggle_cfg.get('fnspid_input_path',
                                           '/kaggle/input/financial-news-and-stock-price-integration-dataset'))

        if not fnspid_path.exists():
            return pd.DataFrame()

        logger.info(f"Found local FNSPID at {fnspid_path}")

        csv_files = list(fnspid_path.glob('**/*.csv'))
        if not csv_files:
            return pd.DataFrame()

        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, on_bad_lines='skip')
                if len(df) > 0:
                    dfs.append(df)
            except Exception:
                continue

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined)} articles from {len(dfs)} local FNSPID files")

        # Standardize column names
        column_mappings = {
            'headline': 'text', 'headlines': 'text', 'title': 'text',
            'news': 'text', 'article': 'text',
            'published': 'date', 'publish_date': 'date', 'timestamp': 'date',
            'ticker': 'symbol', 'stock': 'symbol', 'company': 'symbol'
        }
        for col in combined.columns:
            if col.lower() in column_mappings:
                combined.rename(columns={col: column_mappings[col.lower()]}, inplace=True)

        return combined

    def load_fnspid(self) -> pd.DataFrame:
        """Load FNSPID dataset from local path (Kaggle) or Hugging Face."""
        logger.info("Loading FNSPID dataset for feasibility audit...")

        # Try local path first (Kaggle or manual download)
        local_df = self._load_fnspid_local()
        if not local_df.empty:
            return local_df

        logger.info("Local FNSPID not found, trying Hugging Face...")

        try:
            from datasets import load_dataset

            hf_config = self.config['huggingface']
            dataset = load_dataset(
                hf_config['dataset'],
                cache_dir=hf_config.get('cache_dir', 'data/hf_cache'),
            )

            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                split_name = list(dataset.keys())[0]
                df = dataset[split_name].to_pandas()

            logger.info(f"Loaded {len(df)} total articles from FNSPID")
            return df

        except Exception as e:
            logger.error(f"Failed to load FNSPID: {e}")
            return pd.DataFrame()

    def count_articles_per_ticker(self, df: pd.DataFrame) -> Dict[str, dict]:
        """
        Count unique articles mentioning each ticker.
        Searches both ticker symbols and company name keywords.
        """
        start, end = self._compute_dates()
        logger.info(f"Audit period: {start.date()} to {end.date()}")

        # Identify text column
        text_col = None
        for candidate in ['text', 'headline', 'title', 'news', 'Article']:
            if candidate in df.columns:
                text_col = candidate
                break

        if text_col is None:
            logger.error(f"No text column found. Available columns: {list(df.columns)}")
            return {}

        # Date filtering
        date_col = None
        for candidate in ['date', 'Date', 'published', 'timestamp']:
            if candidate in df.columns:
                date_col = candidate
                break

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df_filtered = df[(df[date_col] >= start) & (df[date_col] <= end)].copy()
            logger.info(f"Articles in date range: {len(df_filtered)}")
        else:
            df_filtered = df.copy()
            logger.warning("No date column found - auditing all articles")

        # Ticker-to-keywords mapping for broader matching
        ticker_keywords = {
            'SMR': ['SMR', 'NuScale', 'small modular reactor'],
            'LEU': ['LEU', 'Centrus', 'uranium enrichment'],
            'LTBR': ['LTBR', 'Lightbridge', 'nuclear fuel technology'],
            'NXE': ['NXE', 'NexGen', 'uranium mining'],
            'NNE': ['NNE', 'Nano Nuclear', 'nano nuclear energy'],
            'LAC': ['LAC', 'Lithium Americas', 'lithium mining'],
            'CCJ': ['CCJ', 'Cameco', 'uranium producer'],
            'CEG': ['CEG', 'Constellation Energy', 'nuclear utility'],
            'BWXT': ['BWXT', 'BWX Technologies', 'nuclear components'],
        }

        results = {}
        for ticker in self.all_tickers:
            keywords = ticker_keywords.get(ticker, [ticker])
            pattern = '|'.join(keywords)

            matches = df_filtered[
                df_filtered[text_col].str.contains(pattern, case=False, na=False, regex=True)
            ]

            unique_articles = matches.drop_duplicates(subset=[text_col])
            group = 'core_set' if ticker in self.core_set else 'benchmark_set'

            results[ticker] = {
                'count': len(unique_articles),
                'group': group,
                'passes_threshold': len(unique_articles) >= self.min_articles,
                'keywords_used': keywords,
            }

            status = "PASS" if results[ticker]['passes_threshold'] else "FAIL"
            logger.info(f"  {ticker} ({group}): {len(unique_articles)} articles [{status}]")

        return results

    def generate_report(self, results: Dict[str, dict]) -> dict:
        """Generate audit report with recommendations"""
        passing = {t: r for t, r in results.items() if r['passes_threshold']}
        failing = {t: r for t, r in results.items() if not r['passes_threshold']}

        report = {
            'audit_date': datetime.now().isoformat(),
            'min_articles_threshold': self.min_articles,
            'total_tickers': len(results),
            'passing_tickers': len(passing),
            'failing_tickers': len(failing),
            'results': {},
            'recommendations': []
        }

        for ticker, data in results.items():
            report['results'][ticker] = {
                'article_count': data['count'],
                'group': data['group'],
                'status': 'PASS' if data['passes_threshold'] else 'FAIL'
            }

        for ticker, data in failing.items():
            if self.config['phase0'].get('supplement_via_scraping', True):
                report['recommendations'].append(
                    f"{ticker}: Supplement with Yahoo Finance scraping "
                    f"(only {data['count']} articles, need {self.min_articles})"
                )
            else:
                report['recommendations'].append(
                    f"{ticker}: Consider removing from study "
                    f"(only {data['count']} articles, need {self.min_articles})"
                )

        if not failing:
            report['recommendations'].append("All tickers pass the minimum article threshold.")

        return report

    def save_report(self, report: dict):
        """Save audit report to YAML"""
        report_path = Path(self.config['phase0'].get(
            'audit_report_path', 'results/feasibility_audit.yaml'
        ))
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Audit report saved to {report_path}")

    def run_audit(self) -> dict:
        """Run the complete feasibility audit"""
        logger.info("=" * 70)
        logger.info("PHASE 0: DATA FEASIBILITY AUDIT")
        logger.info(f"Minimum articles per ticker: {self.min_articles}")
        logger.info("=" * 70)

        df = self.load_fnspid()
        if df.empty:
            logger.error("Cannot run audit - FNSPID dataset unavailable")
            return {}

        results = self.count_articles_per_ticker(df)
        report = self.generate_report(results)
        self.save_report(report)

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("AUDIT SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Passing: {report['passing_tickers']}/{report['total_tickers']}")
        logger.info(f"Failing: {report['failing_tickers']}/{report['total_tickers']}")

        if report['recommendations']:
            logger.info("\nRecommendations:")
            for rec in report['recommendations']:
                logger.info(f"  - {rec}")

        logger.info("=" * 70)
        return report


def main():
    auditor = FeasibilityAuditor()
    auditor.run_audit()


if __name__ == "__main__":
    main()
