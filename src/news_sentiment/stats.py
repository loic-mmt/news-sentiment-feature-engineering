"""Descriptive statistics and supervised evaluation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import TypedDict, cast

import pandas as pd

from .results import (
    PREDICTION_COLUMNS,
    SENTIMENT_LABELS,
    EvaluationReport,
    SentimentLabel,
)

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
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")

    duplicate_columns = [
        column
        for position, column in enumerate(predictions.columns)
        if column in predictions.columns[:position]
    ]
    if duplicate_columns:
        raise ValueError(f"duplicate prediction columns: {duplicate_columns}")

    missing_columns = [
        column for column in PREDICTION_COLUMNS if column not in predictions.columns
    ]
    if missing_columns:
        raise ValueError(f"missing prediction columns: {missing_columns}")
    if predictions.empty:
        raise ValueError("predictions must not be empty")

    required_data = predictions[list(PREDICTION_COLUMNS)]
    numeric_columns = (
        "confidence",
        "p_negative",
        "p_neutral",
        "p_positive",
        "sentiment_score",
    )

    for row_number, row in enumerate(
        required_data.itertuples(index=False, name=None)
    ):
        text, label, *raw_numeric_values = row

        if not isinstance(text, str):
            raise ValueError(
                f"row {row_number}: text must be a string, "
                f"got {type(text).__name__}"
            )
        if not text.strip():
            raise ValueError(f"row {row_number}: text must not be empty or blank")

        if not isinstance(label, str) or label not in SENTIMENT_LABELS:
            raise ValueError(
                f"row {row_number}: label must be one of {list(SENTIMENT_LABELS)}, "
                f"got {label!r}"
            )

        numeric_values: dict[str, float] = {}
        for column, value in zip(
            numeric_columns, raw_numeric_values, strict=True
        ):
            if isinstance(value, (bool, str, bytes)):
                raise ValueError(
                    f"row {row_number}: {column} must be a real number, "
                    f"got {value!r}"
                )
            try:
                numeric_value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"row {row_number}: {column} must be a real number"
                ) from exc
            if not math.isfinite(numeric_value):
                raise ValueError(f"row {row_number}: {column} must be finite")
            numeric_values[column] = numeric_value

        confidence = numeric_values["confidence"]
        probabilities = {
            "negative": numeric_values["p_negative"],
            "neutral": numeric_values["p_neutral"],
            "positive": numeric_values["p_positive"],
        }
        sentiment_score = numeric_values["sentiment_score"]

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"row {row_number}: confidence must be between 0 and 1")
        for sentiment, probability in probabilities.items():
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"row {row_number}: p_{sentiment} must be between 0 and 1"
                )
        if not -1.0 <= sentiment_score <= 1.0:
            raise ValueError(
                f"row {row_number}: sentiment_score must be between -1 and 1"
            )

        probability_sum = sum(probabilities.values())
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError(
                f"row {row_number}: probabilities must sum to 1, "
                f"got {probability_sum:.8f}"
            )

        winning_label = max(SENTIMENT_LABELS, key=probabilities.__getitem__)
        expected_confidence = probabilities[winning_label]
        if label != winning_label:
            raise ValueError(
                f"row {row_number}: label {label!r} does not match winning "
                f"label {winning_label!r}"
            )
        if not math.isclose(
            confidence, expected_confidence, rel_tol=0.0, abs_tol=1e-5
        ):
            raise ValueError(
                f"row {row_number}: confidence must equal winning probability "
                f"{expected_confidence:.8f}"
            )

        expected_score = probabilities["positive"] - probabilities["negative"]
        if not math.isclose(
            sentiment_score, expected_score, rel_tol=0.0, abs_tol=1e-5
        ):
            raise ValueError(
                f"row {row_number}: sentiment_score must equal "
                f"p_positive - p_negative ({expected_score:.8f})"
            )


def summarize(
    predictions: pd.DataFrame,
    *,
    confidence_threshold: float = 0.60,
) -> SentimentSummary:
    """Compute the essential descriptive statistics for a prediction batch."""
    validate_prediction_frame(predictions)

    if isinstance(confidence_threshold, (bool, str, bytes)):
        raise TypeError("confidence_threshold must be a real number")
    try:
        threshold = float(confidence_threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("confidence_threshold must be a real number") from exc
    if not math.isfinite(threshold):
        raise ValueError("confidence_threshold must be finite")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")

    label_counts: dict[SentimentLabel, int] = {
        label: 0 for label in SENTIMENT_LABELS
    }
    scores: list[float] = []
    confidences: list[float] = []

    summary_data = predictions[["label", "sentiment_score", "confidence"]]
    for label, score, confidence in summary_data.itertuples(index=False, name=None):
        label_counts[label] += 1
        scores.append(float(score))
        confidences.append(float(confidence))

    n = len(predictions)
    label_proportions: dict[SentimentLabel, float] = {
        label: count / n for label, count in label_counts.items()
    }
    low_confidence_rate = sum(
        confidence < threshold for confidence in confidences
    ) / n

    summary: SentimentSummary = {
        "n": n,
        "label_counts": label_counts,
        "label_proportions": label_proportions,
        "mean_score": float(sum(scores) / n),
        "median_score": float(statistics.median(scores)),
        "mean_confidence": float(sum(confidences) / n),
        "confidence_threshold": threshold,
        "low_confidence_rate": float(low_confidence_rate),
    }
    return summary


def evaluate(
    predictions: pd.DataFrame,
    y_true: Iterable[str],
) -> EvaluationReport:
    """Compare predicted labels with known labels using sklearn metrics."""
    validate_prediction_frame(predictions)

    if isinstance(y_true, (str, bytes)):
        raise TypeError("y_true must be an iterable of sentiment labels")
    try:
        raw_true_labels = list(y_true)
    except TypeError as exc:
        raise TypeError("y_true must be an iterable of sentiment labels") from exc

    if len(raw_true_labels) != len(predictions):
        raise ValueError(
            "length mismatch between y_true and predictions: "
            f"expected {len(predictions)}, got {len(raw_true_labels)}"
        )

    normalized_true_labels: list[SentimentLabel] = []
    for position, raw_label in enumerate(raw_true_labels):
        if not isinstance(raw_label, str):
            raise ValueError(
                f"y_true[{position}] must be a string, "
                f"got {type(raw_label).__name__}"
            )
        normalized_label = raw_label.strip().casefold()
        if normalized_label not in SENTIMENT_LABELS:
            raise ValueError(
                f"y_true[{position}] must be one of {list(SENTIMENT_LABELS)}, "
                f"got {raw_label!r}"
            )
        normalized_true_labels.append(cast(SentimentLabel, normalized_label))

    predicted_labels = [
        label
        for (label,) in predictions[["label"]].itertuples(index=False, name=None)
    ]

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    accuracy = float(accuracy_score(normalized_true_labels, predicted_labels))
    macro_f1 = float(
        f1_score(
            normalized_true_labels,
            predicted_labels,
            labels=SENTIMENT_LABELS,
            average="macro",
            zero_division=0,
        )
    )
    report_data = classification_report(
        normalized_true_labels,
        predicted_labels,
        labels=SENTIMENT_LABELS,
        target_names=SENTIMENT_LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        normalized_true_labels,
        predicted_labels,
        labels=SENTIMENT_LABELS,
    )

    metric_columns = ("precision", "recall", "f1-score", "support")
    per_class_data = {
        label: {
            metric: (
                int(report_data[label][metric])
                if metric == "support"
                else float(report_data[label][metric])
            )
            for metric in metric_columns
        }
        for label in SENTIMENT_LABELS
    }
    per_class = pd.DataFrame.from_dict(
        per_class_data,
        orient="index",
        columns=metric_columns,
    )
    per_class.index.name = "label"

    return EvaluationReport(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_matrix=matrix,
        labels=SENTIMENT_LABELS,
    )
