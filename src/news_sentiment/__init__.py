"""Public API for FinBERT-powered financial news sentiment features."""

from .analyzer import DEFAULT_MODEL, SentimentAnalyzer
from .alpha_vantage import AlphaVantageError, fetch_alpha_vantage
from .coverage import save_coverage, validate_coverage
from .export import export_sentiment_features
from .features import attach_sentiment, build_sentiment_features
from .history import import_historical_news
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
    "export_sentiment_features",
    "fetch_alpha_vantage",
    "fetch_rss",
    "import_historical_news",
    "plot_confusion_matrix",
    "plot_labels",
    "plot_score_distribution",
    "plot_timeline",
    "save_news",
    "save_coverage",
    "summarize",
    "validate_coverage",
]
