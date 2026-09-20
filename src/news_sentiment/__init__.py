"""Public API for FinBERT-powered financial news sentiment features."""

from .analyzer import DEFAULT_MODEL, SentimentAnalyzer
from .alpha_vantage import AlphaVantageError, fetch_alpha_vantage
from .features import attach_sentiment, build_sentiment_features
from .news import fetch_rss, save_news
from .plots import (
    plot_confusion_matrix,
    plot_labels,
    plot_score_distribution,
    plot_timeline,
)
from .results import EvaluationReport, Prediction, SentimentLabel
from .quota import AlphaVantageQuota, QuotaExceededError, QuotaStatus
from .stats import SentimentSummary, evaluate, summarize

__all__ = [
    "DEFAULT_MODEL",
    "AlphaVantageError",
    "AlphaVantageQuota",
    "EvaluationReport",
    "Prediction",
    "QuotaExceededError",
    "QuotaStatus",
    "SentimentAnalyzer",
    "SentimentLabel",
    "SentimentSummary",
    "attach_sentiment",
    "build_sentiment_features",
    "evaluate",
    "fetch_alpha_vantage",
    "fetch_rss",
    "plot_confusion_matrix",
    "plot_labels",
    "plot_score_distribution",
    "plot_timeline",
    "save_news",
    "summarize",
]
