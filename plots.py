"""Small, composable Matplotlib visualizations."""

from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes

from .results import EvaluationReport


def _get_ax(ax: Axes | None, *, figsize: tuple[float, float]) -> Axes:
    """Return the supplied axis or create a new one."""
    # TODO: create a figure/axis with pyplot only when ax is None.
    raise NotImplementedError


def plot_labels(
    predictions: pd.DataFrame,
    *,
    proportions: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot label counts, or proportions, in stable sentiment order."""
    # TODO: validate predictions and draw negative/neutral/positive bars.
    # TODO: label the axes and return ax without calling plt.show().
    raise NotImplementedError


def plot_score_distribution(
    predictions: pd.DataFrame,
    *,
    bins: int = 30,
    ax: Axes | None = None,
) -> Axes:
    """Plot a histogram of ``sentiment_score`` with a zero reference line."""
    # TODO: validate predictions/bins, draw the histogram and zero line.
    raise NotImplementedError


def plot_timeline(
    predictions: pd.DataFrame,
    *,
    date_col: str,
    freq: str = "D",
    ax: Axes | None = None,
) -> Axes:
    """Plot mean sentiment score resampled over time."""
    # TODO: parse date_col, reject invalid dates, sort and resample with freq.
    # TODO: plot the aggregated score and return ax.
    raise NotImplementedError


def plot_confusion_matrix(
    report: EvaluationReport,
    *,
    normalize: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot raw counts or row-normalized values from an evaluation report."""
    # TODO: validate matrix shape and optionally normalize rows safely.
    # TODO: draw the matrix, annotations and stable class labels.
    raise NotImplementedError

