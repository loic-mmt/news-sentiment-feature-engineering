"""Tests for prediction validation and descriptive metrics."""

import pandas as pd
import pytest

from news_sentiment import evaluate, summarize
from news_sentiment.stats import validate_prediction_frame


@pytest.fixture
def predictions():
    return pd.DataFrame(
        [
            ["bad", "negative", 0.7, 0.7, 0.2, 0.1, -0.6],
            ["good", "positive", 0.8, 0.1, 0.1, 0.8, 0.7],
        ],
        columns=[
            "text", "label", "confidence", "p_negative", "p_neutral",
            "p_positive", "sentiment_score",
        ],
    )


def test_summary_and_evaluation(predictions):
    original = predictions.copy(deep=True)
    validate_prediction_frame(predictions)
    summary = summarize(predictions)
    assert summary["n"] == 2
    assert summary["label_counts"] == {"negative": 1, "neutral": 0, "positive": 1}
    assert summary["mean_score"] == pytest.approx(0.05)
    report = evaluate(predictions, (label for label in ["negative", "neutral"]))
    assert report.accuracy == pytest.approx(0.5)
    assert report.confusion_matrix.shape == (3, 3)
    assert list(report.per_class.index) == ["negative", "neutral", "positive"]
    pd.testing.assert_frame_equal(predictions, original)


@pytest.mark.parametrize("column,value", [
    ("text", " "),
    ("label", "unknown"),
    ("confidence", float("nan")),
    ("p_positive", 2.0),
    ("sentiment_score", -0.1),
])
def test_invalid_prediction_rows(predictions, column, value):
    predictions.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_prediction_frame(predictions)


def test_invalid_true_labels(predictions):
    with pytest.raises(ValueError, match="length mismatch"):
        evaluate(predictions, ["negative"])
    with pytest.raises(ValueError, match=r"y_true\[1\]"):
        evaluate(predictions, ["negative", "unknown"])
