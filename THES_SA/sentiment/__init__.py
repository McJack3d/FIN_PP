"""
Sentiment Analysis Module (Phase 2)
FinBERT scoring and sentiment feature engineering.
"""

from .scorer import SentimentScorer
from .features import SentimentFeatureEngineer

__all__ = ['SentimentScorer', 'SentimentFeatureEngineer']
