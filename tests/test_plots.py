"""Non-interactive tests for the four plotting helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from news_sentiment import (
    evaluate,
    plot_confusion_matrix,
    plot_labels,
    plot_score_distribution,
    plot_timeline,
)


def test_plots_return_axes_without_mutating_input():
    predictions = pd.DataFrame(
        [["good", "positive", 0.8, 0.1, 0.1, 0.8, 0.7, "2026-01-01"]],
        columns=[
            "text", "label", "confidence", "p_negative", "p_neutral",
            "p_positive", "sentiment_score", "published_at",
        ],
    )
    original = predictions.copy(deep=True)
    report = evaluate(predictions, ["positive"])
    axes = [
        plot_labels(predictions),
        plot_score_distribution(predictions),
        plot_timeline(predictions, date_col="published_at"),
        plot_confusion_matrix(report, normalize=True),
    ]
    assert all(ax.figure is not None for ax in axes)
    assert len(axes[0].patches) == 3
    pd.testing.assert_frame_equal(predictions, original)
    plt.close("all")
