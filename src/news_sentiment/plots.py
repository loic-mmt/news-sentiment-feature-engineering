"""Small, composable Matplotlib visualizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .results import SENTIMENT_LABELS, EvaluationReport
from .stats import validate_prediction_frame

# Palette stable dans toute la librairie : rouge, gris, vert.
SENTIMENT_COLORS: dict[str, str] = {
    "negative": "#C44E52",
    "neutral": "#8C8C8C",
    "positive": "#55A868",
}

# Ordre conseillé : _get_ax -> plot_labels -> plot_score_distribution ->
# plot_confusion_matrix -> plot_timeline. Aucun tracé ne doit appeler show/savefig.


def _get_ax(ax: Axes | None, *, figsize: tuple[float, float]) -> Axes:
    """Return the supplied axis or create a new one."""
    if ax is not None:
        if not isinstance(ax, Axes):
            raise TypeError(f"ax must be a matplotlib Axes, got {type(ax).__name__}")
        return ax

    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_labels(
    predictions: pd.DataFrame,
    *,
    proportions: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot label counts, or proportions, in stable sentiment order."""
    validate_prediction_frame(predictions)
    if not isinstance(proportions, bool):
        raise TypeError("proportions must be a boolean")

    counts = predictions["label"].value_counts().reindex(
        SENTIMENT_LABELS,
        fill_value=0,
    )
    values = counts.to_numpy(dtype=float)
    if proportions:
        values = values / len(predictions)

    ax = _get_ax(ax, figsize=(7, 4))
    ax.bar(
        SENTIMENT_LABELS,
        values,
        color=[SENTIMENT_COLORS[label] for label in SENTIMENT_LABELS],
    )
    ax.set_title("Sentiment label distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Proportion" if proportions else "Count")
    return ax


def plot_score_distribution(
    predictions: pd.DataFrame,
    *,
    bins: int = 30,
    ax: Axes | None = None,
) -> Axes:
    """Plot a histogram of ``sentiment_score`` with a zero reference line."""
    validate_prediction_frame(predictions)
    if isinstance(bins, bool) or not isinstance(bins, int):
        raise TypeError("bins must be an integer")
    if bins < 1:
        raise ValueError("bins must be greater than or equal to 1")

    ax = _get_ax(ax, figsize=(8, 4))
    ax.hist(
        predictions["sentiment_score"].to_numpy(dtype=float),
        bins=bins,
        range=(-1.0, 1.0),
        color="#4C72B0",
        alpha=0.8,
        edgecolor="white",
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(-1.0, 1.0)
    ax.set_title("Sentiment score distribution")
    ax.set_xlabel("Sentiment score")
    ax.set_ylabel("Count")
    return ax


def plot_timeline(
    predictions: pd.DataFrame,
    *,
    date_col: str,
    freq: str = "D",
    ax: Axes | None = None,
) -> Axes:
    """Plot mean sentiment score resampled over time."""
    validate_prediction_frame(predictions)
    if not isinstance(date_col, str):
        raise TypeError("date_col must be a string")
    if not date_col.strip():
        raise ValueError("date_col must not be empty or blank")
    if date_col not in predictions.columns:
        raise ValueError(f"date column {date_col!r} is missing from predictions")
    if not isinstance(freq, str):
        raise TypeError("freq must be a string")
    if not freq.strip():
        raise ValueError("freq must not be empty or blank")

    timeline = predictions[[date_col, "sentiment_score"]].copy()
    converted_dates = pd.to_datetime(timeline[date_col], errors="coerce", utc=True)
    invalid_date_count = int(converted_dates.isna().sum())
    if invalid_date_count:
        raise ValueError(
            f"date column {date_col!r} contains {invalid_date_count} invalid value(s)"
        )
    timeline[date_col] = converted_dates

    try:
        aggregated = (
            timeline.sort_values(date_col)
            .set_index(date_col)["sentiment_score"]
            .resample(freq.strip())
            .mean()
            .dropna()
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid resampling frequency {freq!r}") from exc

    ax = _get_ax(ax, figsize=(10, 4))
    ax.plot(
        aggregated.index,
        aggregated.to_numpy(dtype=float),
        color="#4C72B0",
        linewidth=1.5,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Mean sentiment over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean sentiment score")
    return ax


def plot_confusion_matrix(
    report: EvaluationReport,
    *,
    normalize: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot raw counts or row-normalized values from an evaluation report."""
    if not isinstance(report, EvaluationReport):
        raise TypeError("report must be an EvaluationReport")
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a boolean")

    try:
        matrix = np.array(report.confusion_matrix, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("confusion matrix must contain numeric values") from exc
    expected_shape = (len(report.labels), len(report.labels))
    if matrix.shape != expected_shape:
        raise ValueError(
            f"confusion matrix must have shape {expected_shape}, got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("confusion matrix must contain only finite values")
    if (matrix < 0).any():
        raise ValueError("confusion matrix must not contain negative values")

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix),
            where=row_sums != 0,
        )

    ax = _get_ax(ax, figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    ax.figure.colorbar(image, ax=ax)
    positions = range(len(report.labels))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(report.labels)
    ax.set_yticklabels(report.labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized confusion matrix" if normalize else "Confusion matrix")

    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = float(matrix[row_index, column_index])
            text = f"{value:.2f}" if normalize else f"{value:g}"
            ax.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    return ax
