"""Tests for result serialization."""

from news_sentiment.results import PREDICTION_COLUMNS, Prediction


def test_prediction_to_record_preserves_values_and_order():
    probabilities = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
    prediction = Prediction("text", "positive", 0.7, probabilities, 0.6)
    record = prediction.to_record()
    assert list(record) == list(PREDICTION_COLUMNS)
    assert record == {
        "text": "text",
        "label": "positive",
        "confidence": 0.7,
        "p_negative": 0.1,
        "p_neutral": 0.2,
        "p_positive": 0.7,
        "sentiment_score": 0.6,
    }
    assert probabilities == {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
