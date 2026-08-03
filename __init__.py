"""Public API for the news sentiment mini-library."""

from .analyzer import DEFAULT_MODEL, SentimentAnalyzer
from .plots import (
    plot_confusion_matrix,
    plot_labels,
    plot_score_distribution,
    plot_timeline,
)
from .results import EvaluationReport, Prediction, SentimentLabel
from .stats import SentimentSummary, evaluate, summarize

__all__ = [
    "DEFAULT_MODEL",
    "EvaluationReport",
    "Prediction",
    "SentimentAnalyzer",
    "SentimentLabel",
    "SentimentSummary",
    "evaluate",
    "plot_confusion_matrix",
    "plot_labels",
    "plot_score_distribution",
    "plot_timeline",
    "summarize",
]

