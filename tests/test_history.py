"""Historical availability must be explicit and supported by a source reference."""

import pandas as pd
import pytest

from news_sentiment import (
    attach_sentiment, export_sentiment_features, import_historical_news, save_news,
)


def historical_row() -> pd.DataFrame:
    return pd.DataFrame([{
        "news_id": "vendor-1", "ticker": "AAPL", "source": "vendor-feed",
        "published_at": pd.Timestamp("2026-09-18T08:00:00Z"),
        "available_at": pd.Timestamp("2026-09-18T08:03:00Z"),
        "availability_kind": "provider_first_seen",
        "availability_reference": "vendor-record:123:first_seen_at",
        "url": "https://example.test/1", "title": "Apple news", "text": "Apple news",
    }])


def test_historical_import_requires_evidence_and_utc_timestamps():
    row = historical_row()
    original = row.copy(deep=True)
    imported = import_historical_news(row)
    assert imported.loc[0, "available_at"] == pd.Timestamp("2026-09-18T08:03:00Z")
    assert imported.loc[0, "availability_kind"] == "provider_first_seen"
    pd.testing.assert_frame_equal(row, original)

    for column in ("availability_kind", "availability_reference"):
        with pytest.raises(ValueError, match="missing historical availability columns"):
            import_historical_news(row.drop(columns=column))
    bad = row.copy()
    bad["availability_kind"] = "pipeline_observed"
    with pytest.raises(ValueError, match="availability_kind"):
        import_historical_news(bad)
    bad = row.copy()
    bad["availability_reference"] = " "
    with pytest.raises(ValueError, match="availability_reference"):
        import_historical_news(bad)
    bad = row.copy()
    bad["available_at"] = "2026-09-18 08:03:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        import_historical_news(bad)
    bad = row.copy()
    bad["available_at"] = pd.Timestamp("2026-09-18T07:00:00Z")
    with pytest.raises(ValueError, match="precedes published_at"):
        import_historical_news(bad)


def test_replay_cannot_backdate_persisted_first_observation(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "news.parquet"
    first = historical_row()
    first["availability_kind"] = "archive_first_seen"
    first["available_at"] = pd.Timestamp("2026-09-18T10:00:00Z")
    save_news(import_historical_news(first), path)
    replay = historical_row()
    stored = save_news(import_historical_news(replay), path)
    assert stored.loc[0, "available_at"] == pd.Timestamp("2026-09-18T10:00:00Z")
    assert stored.loc[0, "availability_kind"] == "archive_first_seen"


def test_historical_import_scores_and_exports_with_strict_cutoff(tmp_path):
    pytest.importorskip("pyarrow")

    class Analyzer:
        def predict(self, texts):
            return pd.DataFrame([{
                "text": text, "label": "positive", "confidence": 0.8,
                "p_negative": 0.1, "p_neutral": 0.1, "p_positive": 0.8,
                "sentiment_score": 0.7,
            } for text in texts])

    scored = attach_sentiment(import_historical_news(historical_row()), Analyzer())
    points = pd.DataFrame({
        "ticker": ["AAPL", "AAPL"],
        "decision_at": [
            pd.Timestamp("2026-09-18T08:03:00Z"),
            pd.Timestamp("2026-09-18T08:04:00Z"),
        ],
    })
    features = export_sentiment_features(
        scored, points, tmp_path / "historical-features.parquet",
        checkpoint="finbert@immutable-revision", include_at_cutoff=False,
    )
    assert features["news_count"].tolist() == [0, 1]
    assert features["coverage_status"].tolist() == ["unknown", "unknown"]
    assert features.loc[1, "last_contributing_available_at"] == pd.Timestamp(
        "2026-09-18T08:03:00Z"
    )
