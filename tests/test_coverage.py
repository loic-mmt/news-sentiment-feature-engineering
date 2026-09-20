"""Coverage is a separate, conservative, point-in-time journal."""

import pandas as pd
import pytest

from news_sentiment import build_sentiment_features, save_coverage, validate_coverage


def journal() -> pd.DataFrame:
    return pd.DataFrame([{
        "source": "feed-a", "ticker": None,
        "start_at": pd.Timestamp("2026-09-18T04:00:00Z"),
        "end_at": pd.Timestamp("2026-09-18T10:00:00Z"),
        "recorded_at": pd.Timestamp("2026-09-18T09:30:00Z"),
        "status": "covered", "evidence_reference": "snapshot:feed-a:1",
    }])


def points() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAPL"],
        "decision_at": [pd.Timestamp("2026-09-18T10:00:00Z")],
    })


def empty_scored_news() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "news_id", "ticker", "available_at", "text", "label", "confidence",
        "p_negative", "p_neutral", "p_positive", "sentiment_score",
    ])


def test_coverage_status_requires_all_sources_and_full_window():
    news = empty_scored_news()
    base = journal()
    result = build_sentiment_features(
        news, points(), lookback="6h", short_lookback="1h", include_at_cutoff=False,
        coverage=base, required_sources=["feed-a"],
    )
    assert result.loc[0, "news_count"] == 0
    assert result.loc[0, "coverage_status"] == "covered"
    assert pd.isna(result.loc[0, "last_contributing_available_at"])

    missing = build_sentiment_features(
        news, points(), lookback="6h", short_lookback="1h", include_at_cutoff=False,
        coverage=base, required_sources=["feed-a", "feed-b"],
    )
    assert missing.loc[0, "coverage_status"] == "unknown"

    partial = base.copy()
    partial["start_at"] = pd.Timestamp("2026-09-18T05:00:00Z")
    assert build_sentiment_features(
        news, points(), lookback="6h", short_lookback="1h", include_at_cutoff=False,
        coverage=partial, required_sources=["feed-a"],
    ).loc[0, "coverage_status"] == "unknown"

    incomplete = base.copy()
    incomplete["status"] = "incomplete"
    assert build_sentiment_features(
        news, points(), lookback="6h", short_lookback="1h", include_at_cutoff=False,
        coverage=incomplete, required_sources=["feed-a"],
    ).loc[0, "coverage_status"] == "incomplete"


def test_late_or_other_ticker_coverage_cannot_rewrite_past():
    base = journal()
    late = base.copy()
    late["recorded_at"] = pd.Timestamp("2026-09-18T11:00:00Z")
    other = base.copy()
    other["ticker"] = "MSFT"
    for log in (late, other):
        result = build_sentiment_features(
            empty_scored_news(), points(), lookback="6h", short_lookback="1h", include_at_cutoff=False,
            coverage=log, required_sources=["feed-a"],
        )
        assert result.loc[0, "coverage_status"] == "unknown"


def test_later_correction_changes_only_future_coverage():
    base = journal()
    base["end_at"] = pd.Timestamp("2026-09-18T11:00:00Z")
    gap = base.copy()
    gap["start_at"] = pd.Timestamp("2026-09-18T08:00:00Z")
    gap["end_at"] = pd.Timestamp("2026-09-18T09:00:00Z")
    gap["recorded_at"] = pd.Timestamp("2026-09-18T09:40:00Z")
    gap["status"] = "incomplete"
    repair = gap.copy()
    repair["recorded_at"] = pd.Timestamp("2026-09-18T10:15:00Z")
    repair["status"] = "covered"
    log = pd.concat([base, gap, repair], ignore_index=True)
    decision_points = pd.DataFrame({
        "ticker": ["AAPL", "AAPL"],
        "decision_at": [
            pd.Timestamp("2026-09-18T10:00:00Z"),
            pd.Timestamp("2026-09-18T10:30:00Z"),
        ],
    })
    features = build_sentiment_features(
        empty_scored_news(), decision_points,
        lookback="6h", short_lookback="1h", include_at_cutoff=False,
        coverage=log, required_sources=["feed-a"],
    )
    assert features["coverage_status"].tolist() == ["incomplete", "covered"]


def test_coverage_validation_and_idempotent_storage(tmp_path):
    pytest.importorskip("pyarrow")
    base = journal()
    path = tmp_path / "coverage.parquet"
    assert len(save_coverage(base, path)) == 1
    assert len(save_coverage(base, path)) == 1
    bad = base.copy()
    bad["status"] = "unknown"
    with pytest.raises(ValueError, match="status"):
        validate_coverage(bad)
    bad = base.copy()
    bad["recorded_at"] = "2026-09-18 09:30:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_coverage(bad)
