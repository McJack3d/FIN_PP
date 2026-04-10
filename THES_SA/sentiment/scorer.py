"""
FinBERT Sentiment Scorer (Phase 2)
Scores each news headline with (positive, negative, neutral) probabilities.
Outputs scored dataset to data/processed/news_scored.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import List, Optional
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentScorer:
    """Scores news headlines using FinBERT"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_name = self.config['sentiment']['model']
        self.max_length = self.config['sentiment']['max_length']
        self.batch_size = self.config['sentiment']['batch_size']

        self.processed_path = Path(self.config['paths']['processed'])
        self.processed_path.mkdir(parents=True, exist_ok=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None

        logger.info(f"SentimentScorer initialized (device: {self.device})")
        if self.device == 'cuda':
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
            logger.info(f"  VRAM: {vram / 1e9:.1f} GB")

    def load_model(self):
        """Load FinBERT model and tokenizer"""
        if self.model is not None:
            return

        logger.info(f"Loading FinBERT model: {self.model_name}...")

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info("FinBERT model loaded successfully")

    def score_texts(self, texts: List[str]) -> List[dict]:
        """
        Score a list of texts with FinBERT.

        Args:
            texts: List of headline strings

        Returns:
            List of dicts with 'positive', 'negative', 'neutral' probabilities
            and 'sentiment_score' (positive - negative).
        """
        self.load_model()
        results = []

        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Scoring"):
            batch = texts[i:i + self.batch_size]

            # Clean empty/null texts
            batch = [str(t) if t and str(t).strip() else "neutral" for t in batch]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # FinBERT output order: positive, negative, neutral
            probs_np = probs.cpu().numpy()

            for j in range(len(batch)):
                positive = float(probs_np[j][0])
                negative = float(probs_np[j][1])
                neutral = float(probs_np[j][2])

                results.append({
                    'positive': positive,
                    'negative': negative,
                    'neutral': neutral,
                    'sentiment_score': positive - negative,
                })

        return results

    def score_dataframe(self, df: pd.DataFrame,
                        text_column: str = 'cleaned_text') -> pd.DataFrame:
        """
        Score all texts in a DataFrame.

        Args:
            df: DataFrame with a text column
            text_column: Column containing text to score

        Returns:
            DataFrame with added sentiment columns
        """
        logger.info(f"Scoring {len(df)} articles with FinBERT...")

        if df.empty:
            logger.warning("No articles to score - returning empty DataFrame")
            for col in ['positive', 'negative', 'neutral', 'sentiment_score']:
                df[col] = pd.Series(dtype=float)
            return df

        # Use cleaned_text if available, fall back to text
        if text_column not in df.columns:
            if 'text' in df.columns:
                text_column = 'text'
            else:
                raise ValueError(f"No text column found. Available: {list(df.columns)}")

        texts = df[text_column].tolist()
        scores = self.score_texts(texts)

        # Add scores to dataframe
        df_scored = df.copy()
        if scores:
            scores_df = pd.DataFrame(scores)
            for col in scores_df.columns:
                df_scored[col] = scores_df[col].values
        else:
            for col in ['positive', 'negative', 'neutral', 'sentiment_score']:
                df_scored[col] = 0.0

        logger.info(f"Scoring complete. Mean sentiment: {df_scored['sentiment_score'].mean():.4f}")
        logger.info(f"  Positive: {(df_scored['sentiment_score'] > 0.05).sum()} articles")
        logger.info(f"  Negative: {(df_scored['sentiment_score'] < -0.05).sum()} articles")
        logger.info(f"  Neutral:  {(abs(df_scored['sentiment_score']) <= 0.05).sum()} articles")

        return df_scored

    def run_scoring_pipeline(self,
                             input_file: str = 'news_processed.csv',
                             output_file: str = 'news_scored.csv') -> pd.DataFrame:
        """
        Run the full scoring pipeline: load processed news, score, save.

        Args:
            input_file: Processed news CSV filename (in data/processed/)
            output_file: Output scored CSV filename (in data/processed/)

        Returns:
            Scored DataFrame
        """
        logger.info("=" * 60)
        logger.info("PHASE 2a: FINBERT SENTIMENT SCORING")
        logger.info("=" * 60)

        # Load processed news
        input_path = self.processed_path / input_file
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            logger.info("Run Phase 1 (data pipeline) first.")
            return pd.DataFrame()

        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} processed articles")

        # Score
        df_scored = self.score_dataframe(df)

        # Save
        output_path = self.processed_path / output_file
        df_scored.to_csv(output_path, index=False)
        logger.info(f"Saved scored data to {output_path}")

        logger.info("=" * 60)
        logger.info("SENTIMENT SCORING COMPLETE")
        logger.info("=" * 60)

        return df_scored


def main():
    scorer = SentimentScorer()
    scorer.run_scoring_pipeline()


if __name__ == "__main__":
    main()
