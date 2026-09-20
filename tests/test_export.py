"""Feature Parquet and sidecar manifest can be audited together."""

import hashlib
import json

import pandas as pd
import pytest

from news_sentiment import export_sentiment_features


def test_export_writes_schema_checkpoint_inputs_and_parquet_checksum(tmp_path):
    pytest.importorskip("pyarrow")
    scored = pd.DataFrame([{
        "news_id": "n1", "ticker": "AAPL",
        "published_at": pd.Timestamp("2026-09-18T08:00:00Z"),
        "available_at": pd.Timestamp("2026-09-18T09:00:00Z"),
        "source": "feed-a", "url": "https://example.test/1", "title": "Good news",
        "availability_kind": "pipeline_observed",
        "availability_reference": "https://example.test/feed.xml",
        "text": "Good news", "label": "positive", "confidence": 0.8,
        "p_negative": 0.1, "p_neutral": 0.1, "p_positive": 0.8,
        "sentiment_score": 0.7,
    }])
    decisions = pd.DataFrame({
        "ticker": ["AAPL"], "decision_at": [pd.Timestamp("2026-09-18T10:00:00Z")],
    })
    coverage = pd.DataFrame([{
        "source": "feed-a", "ticker": None,
        "start_at": pd.Timestamp("2026-09-17T10:00:00Z"),
        "end_at": pd.Timestamp("2026-09-18T10:00:00Z"),
        "recorded_at": pd.Timestamp("2026-09-18T09:30:00Z"),
        "status": "covered", "evidence_reference": "snapshot:feed-a:1",
    }])
    target = tmp_path / "features.parquet"
    features = export_sentiment_features(
        scored, decisions, target, checkpoint="yiyanghkust/finbert-tone@revision",
        input_identifiers={"news_snapshot": "snapshot-123"}, coverage=coverage,
        required_sources=["feed-a"], include_at_cutoff=False,
    )
    stored = pd.read_parquet(target)
    pd.testing.assert_frame_equal(features, stored)
    assert features.loc[0, "coverage_status"] == "covered"
    assert features.loc[0, "last_contributing_available_at"] == pd.Timestamp("2026-09-18T09:00:00Z")

    manifest = json.loads((tmp_path / "features.manifest.json").read_text())
    assert manifest["schema_version"] == "1.0"
    assert manifest["checkpoint"] == "yiyanghkust/finbert-tone@revision"
    assert manifest["aggregation"]["include_at_cutoff"] is False
    assert manifest["aggregation"]["required_sources"] == ["feed-a"]
    assert manifest["input_identifiers"] == {"news_snapshot": "snapshot-123"}
    assert manifest["inputs"]["scored_news"]["rows"] == 1
    assert manifest["parquet_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


def test_export_requires_checkpoint_before_writing(tmp_path):
    with pytest.raises(ValueError, match="checkpoint"):
        export_sentiment_features(pd.DataFrame(), pd.DataFrame(), tmp_path / "x.parquet", checkpoint="")
    assert not (tmp_path / "x.parquet").exists()


def test_export_refuses_news_without_availability_provenance(tmp_path):
    news = pd.DataFrame([{
        "news_id": "n1", "ticker": "AAPL", "source": "feed-a",
        "published_at": pd.Timestamp("2026-09-18T08:00:00Z"),
        "available_at": pd.Timestamp("2026-09-18T09:00:00Z"),
        "url": "", "title": "Good news", "text": "Good news",
    }])
    with pytest.raises(ValueError, match="availability provenance"):
        export_sentiment_features(
            news, pd.DataFrame(), tmp_path / "x.parquet", checkpoint="finbert@revision"
        )


def test_export_keeps_empty_pair_and_missing_statistics(tmp_path):
    pytest.importorskip("pyarrow")
    empty = pd.DataFrame(columns=[
        "news_id", "published_at", "available_at", "source", "url", "ticker",
        "title", "text", "availability_kind", "availability_reference",
        "label", "confidence", "p_negative", "p_neutral", "p_positive",
        "sentiment_score",
    ])
    points = pd.DataFrame({
        "ticker": ["AAPL"],
        "decision_at": [pd.Timestamp("2026-09-18T10:00:00Z")],
    })
    result = export_sentiment_features(
        empty, points, tmp_path / "empty.parquet", checkpoint="finbert@revision"
    )
    assert result.loc[0, "news_count"] == 0
    assert result.loc[0, "coverage_status"] == "unknown"
    assert pd.isna(result.loc[0, "sentiment_mean"])
