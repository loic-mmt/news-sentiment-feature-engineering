"""Feature alignment and strict point-in-time tests."""

from __future__ import annotations

import pandas as pd
import pytest

from news_sentiment import attach_sentiment, build_sentiment_features


class FakeAnalyzer:
    def __init__(self):
        self.calls = []

    def predict(self, texts):
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            if text.startswith("good") or text.startswith("future"):
                rows.append([text, "positive", 0.8, 0.1, 0.1, 0.8, 0.7])
            else:
                rows.append([text, "negative", 0.7, 0.7, 0.2, 0.1, -0.6])
        return pd.DataFrame(
            rows,
            columns=[
                "text", "label", "confidence", "p_negative", "p_neutral",
                "p_positive", "sentiment_score",
            ],
        )


@pytest.fixture
def news():
    frame = pd.DataFrame(
        [
            ["one", "AAPL", "2026-09-18T08:00:00Z", "good story"],
            ["two", "AAPL", "2026-09-18T09:00:00Z", "bad story"],
            ["three", "AAPL", "2026-09-18T12:00:00Z", "future story"],
        ],
        columns=["news_id", "ticker", "available_at", "text"],
        index=[4, 4, 9],
    )
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    return frame


def test_attach_sentiment_preserves_metadata_index_and_order(news):
    analyzer = FakeAnalyzer()
    original = news.copy(deep=True)
    scored = attach_sentiment(news, analyzer)
    assert analyzer.calls == [["good story", "bad story", "future story"]]
    assert scored.index.tolist() == [4, 4, 9]
    assert scored["news_id"].tolist() == ["one", "two", "three"]
    assert scored["sentiment_score"].tolist() == pytest.approx([0.7, -0.6, 0.7])
    pd.testing.assert_frame_equal(news, original)


def test_features_use_only_available_news_and_keep_empty_pairs(news):
    scored = attach_sentiment(news, FakeAnalyzer())
    points = pd.DataFrame({
        "ticker": ["AAPL", "AAPL", "MSFT"],
        "decision_at": [
            pd.Timestamp("2026-09-18T10:00:00Z"),
            pd.Timestamp("2026-09-18T13:00:00Z"),
            pd.Timestamp("2026-09-18T10:00:00Z"),
        ],
    })
    features = build_sentiment_features(scored, points)
    assert features["ticker"].tolist() == points["ticker"].tolist()
    assert features["news_count"].tolist() == [2, 3, 0]
    assert features.loc[0, "sentiment_mean"] == pytest.approx(0.05)
    assert features.loc[0, "hours_since_last_news"] == pytest.approx(1.0)
    assert features.loc[0, "sentiment_std"] == pytest.approx(0.65)
    assert features.loc[0, "p_positive_mean"] == pytest.approx(0.45)
    assert features.loc[0, "positive_share"] == pytest.approx(0.5)
    assert features.loc[2, "news_count"] == 0
    assert pd.isna(features.loc[2, "sentiment_mean"])

    shorter = build_sentiment_features(scored, points, short_lookback="90min")
    assert shorter.loc[0, "sentiment_momentum"] == pytest.approx(-0.65)
    tiny_half_life = build_sentiment_features(scored, points, half_life="1ns")
    assert tiny_half_life.loc[0, "sentiment_ewm"] == pytest.approx(-0.6)

    moved = scored.copy()
    moved.iloc[2, moved.columns.get_loc("available_at")] = pd.Timestamp("2026-09-18T14:00:00Z")
    moved_features = build_sentiment_features(moved, points)
    pd.testing.assert_series_equal(features.iloc[0], moved_features.iloc[0])


def test_left_boundary_is_excluded_and_news_id_deduplicated(news):
    scored = attach_sentiment(news, FakeAnalyzer())
    duplicate = scored.iloc[[0]].copy()
    duplicate["available_at"] = pd.Timestamp("2026-09-18T09:30:00Z")
    scored = pd.concat([scored, duplicate])
    points = pd.DataFrame({
        "ticker": ["AAPL"],
        "decision_at": [pd.Timestamp("2026-09-19T08:00:00Z")],
    })
    features = build_sentiment_features(scored, points)
    assert features.loc[0, "news_count"] == 2  # 08:00 yesterday is excluded.


def test_rejects_naive_decision_time_and_duplicate_pairs(news):
    scored = attach_sentiment(news, FakeAnalyzer())
    with pytest.raises(ValueError, match="timezone-aware"):
        build_sentiment_features(
            scored,
            pd.DataFrame({"ticker": ["AAPL"], "decision_at": ["2026-09-18 10:00"]}),
        )
    point = pd.DataFrame({
        "ticker": ["AAPL"],
        "decision_at": [pd.Timestamp("2026-09-18T10:00:00Z")],
    })
    with pytest.raises(ValueError, match="duplicate decision point"):
        build_sentiment_features(scored, pd.concat([point, point]))


def test_empty_news_keeps_explicit_decision_grid(news):
    empty = news.iloc[0:0]
    analyzer = FakeAnalyzer()
    scored = attach_sentiment(empty, analyzer)
    assert analyzer.calls == []
    points = pd.DataFrame({
        "ticker": ["AAPL"],
        "decision_at": [pd.Timestamp("2026-09-18T10:00:00Z")],
    })
    features = build_sentiment_features(scored, points)
    assert features.loc[0, "news_count"] == 0
    assert pd.isna(features.loc[0, "hours_since_last_news"])
