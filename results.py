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
        # TODO: flatten probabilities into p_negative, p_neutral and p_positive.
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Metrics produced by an evaluation against known labels."""

    accuracy: float
    macro_f1: float
    per_class: pd.DataFrame
    confusion_matrix: np.ndarray
    labels: tuple[SentimentLabel, ...] = SENTIMENT_LABELS

