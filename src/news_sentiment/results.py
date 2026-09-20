"""Typed result objects returned by the library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd

SentimentLabel: TypeAlias = Literal["negative", "neutral", "positive"]
SENTIMENT_LABELS: tuple[SentimentLabel, ...] = (
    "negative",
    "neutral",
    "positive",
)

# Shared output schema for prediction, statistics, and plotting.
PREDICTION_COLUMNS: tuple[str, ...] = (
    "text",
    "label",
    "confidence",
    "p_negative",
    "p_neutral",
    "p_positive",
    "sentiment_score",
)


@dataclass(frozen=True, slots=True)
class Prediction:
    """Sentiment prediction for a single text.

    ``score`` is defined as ``P(positive) - P(negative)`` and therefore lies
    between -1 and 1.
    """

    text: str
    label: SentimentLabel
    confidence: float
    probabilities: dict[SentimentLabel, float]
    score: float

    def to_record(self) -> dict[str, object]:
        """Convert the prediction to the flat record used by ``predict``."""
        return {
            "text": self.text,
            "label": self.label,
            "confidence": self.confidence,
            "p_negative": self.probabilities["negative"],
            "p_neutral": self.probabilities["neutral"],
            "p_positive": self.probabilities["positive"],
            "sentiment_score": self.score,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Metrics produced by an evaluation against known labels.

    ``per_class`` has labels as its index and ``precision``, ``recall``,
    ``f1-score``, and ``support`` as columns. Both axes of
    ``confusion_matrix`` follow the order in ``labels``.
    """

    accuracy: float
    macro_f1: float
    per_class: pd.DataFrame
    confusion_matrix: np.ndarray
    labels: tuple[SentimentLabel, ...] = SENTIMENT_LABELS
