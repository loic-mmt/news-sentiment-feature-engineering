"""Descriptive statistics and supervised evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import pandas as pd

from .results import EvaluationReport, SentimentLabel


class SentimentSummary(TypedDict):
    """Schema returned by ``summarize``."""

    n: int
    label_counts: dict[SentimentLabel, int]
    label_proportions: dict[SentimentLabel, float]
    mean_score: float
    median_score: float
    mean_confidence: float
    confidence_threshold: float
    low_confidence_rate: float


def validate_prediction_frame(predictions: pd.DataFrame) -> None:
    """Validate the columns and basic value ranges of prediction output."""
    # TODO: check required columns, valid labels, numeric dtypes and finite values.
    # TODO: check probability/score bounds and probability sums.
    raise NotImplementedError


def summarize(
    predictions: pd.DataFrame,
    *,
    confidence_threshold: float = 0.60,
) -> SentimentSummary:
    """Compute the essential descriptive statistics for a prediction batch."""
    # TODO: validate the frame and confidence threshold.
    # TODO: compute counts/proportions and score/confidence aggregates.
    raise NotImplementedError


def evaluate(
    predictions: pd.DataFrame,
    y_true: Sequence[str],
) -> EvaluationReport:
    """Compare predicted labels with known labels using sklearn metrics."""
    # TODO: validate the frame, label values and equal lengths.
    # TODO: compute accuracy, macro-F1, per-class metrics and confusion matrix.
    # TODO: always use the stable negative/neutral/positive label order.
    raise NotImplementedError

