"""Offline Alpha Vantage ingestion and persistent quota tests."""

from __future__ import annotations

import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

import news_sentiment.alpha_vantage as av_module
from news_sentiment import (
    AlphaVantageError,
    AlphaVantageQuota,
    QuotaExceededError,
    fetch_alpha_vantage,
    save_news,
)


PAYLOAD = {
    "items": 3,
    "feed": [
        {
            "title": "<b>Apple and Microsoft rise</b>",
            "url": "https://example.test/story#fragment",
            "time_published": "20260918T080000",
            "summary": "Stocks gain after results.",
            "source": "Market Wire",
            "overall_sentiment_score": 0.42,
            "overall_sentiment_label": "Bullish",
            "ticker_sentiment": [
                {"ticker": "AAPL", "relevance_score": "0.8", "ticker_sentiment_score": "0.5", "ticker_sentiment_label": "Bullish"},
                {"ticker": "MSFT", "relevance_score": "0.4", "ticker_sentiment_score": "0.1", "ticker_sentiment_label": "Somewhat-Bullish"},
            ],
        },
        {"title": "General market news", "summary": "No ticker tags.", "source": "Wire"},
        {"title": "", "summary": "", "ticker_sentiment": [{"ticker": "AAPL"}]},
    ],
}


def test_fetch_normalizes_multiticker_and_preserves_first_observation(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    requests = []

    def fake_open(request, timeout):
        requests.append((parse_qs(urlsplit(request.full_url).query), timeout))
        return BytesIO(json.dumps(PAYLOAD).encode())

    monkeypatch.setattr(av_module, "urlopen", fake_open)
    ledger = tmp_path / "quota.sqlite3"
    first_at = pd.Timestamp("2026-09-18T10:00:00Z")
    news = fetch_alpha_vantage(
        api_key="test-secret", tickers=["AAPL", "MSFT"],
        time_from="20260918T0700", time_to="20260918T0900",
        fetched_at=first_at, quota_path=ledger,
    )
    assert len(news) == 3
    assert news["ticker"].tolist()[:2] == ["AAPL", "MSFT"]
    assert pd.isna(news.loc[2, "ticker"])
    assert news.loc[0, "published_at"] == pd.Timestamp("2026-09-18T08:00:00Z")
    assert news.loc[0, "available_at"] == first_at
    assert news.loc[0, "url"] == "https://example.test/story"
    assert news.loc[0, "text"] == "Apple and Microsoft rise. Stocks gain after results."
    assert news.loc[1, "av_relevance_score"] == pytest.approx(0.4)
    assert news.loc[0, "av_ticker_sentiment_score"] == pytest.approx(0.5)
    assert requests[0][0]["tickers"] == ["AAPL,MSFT"]
    assert requests[0][0]["function"] == ["NEWS_SENTIMENT"]
    assert requests[0][0]["time_from"] == ["20260918T0700"]
    assert requests[0][1] == 10

    stored = save_news(news, tmp_path / "news.parquet")
    later = fetch_alpha_vantage(
        api_key="test-secret", fetched_at=pd.Timestamp("2026-09-18T12:00:00Z"),
        quota_path=ledger,
    )
    stored = save_news(later, tmp_path / "news.parquet")
    assert len(stored) == 3
    assert stored["available_at"].eq(first_at).all()
    assert stored.loc[0, "av_relevance_score"] == pytest.approx(0.8)
    assert AlphaVantageQuota("test-secret", path=ledger).status().day_used == 2
    assert b"test-secret" not in ledger.read_bytes()


def test_quota_persists_and_rolls_windows(monkeypatch, tmp_path):
    now = [1_000_000.0]
    monkeypatch.setattr(AlphaVantageQuota, "_now_epoch", lambda self: now[0])
    path = tmp_path / "usage.sqlite3"
    tracker = AlphaVantageQuota("key-a", path=path, hourly_limit=2, daily_limit=3)
    assert not path.exists()
    assert tracker.status().hour_remaining == 2
    tracker.reserve()
    tracker.reserve()
    with pytest.raises(QuotaExceededError, match="1h: 2/2"):
        tracker.reserve()
    assert AlphaVantageQuota("key-a", path=path, hourly_limit=2, daily_limit=3).status().day_used == 2
    assert AlphaVantageQuota("key-b", path=path, hourly_limit=2, daily_limit=3).status().day_used == 0
    now[0] += 3601
    assert tracker.status().hour_used == 0
    assert tracker.status().day_used == 2
    tracker.reserve()
    with pytest.raises(QuotaExceededError, match="24h: 3/3"):
        tracker.reserve()
    now[0] += 86401
    assert tracker.status().day_remaining == 3
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 3


def test_quota_reservation_is_atomic_across_workers(tmp_path):
    path = tmp_path / "concurrent.sqlite3"

    def reserve_once(_):
        try:
            AlphaVantageQuota("shared-key", path=path, hourly_limit=2).reserve()
            return True
        except QuotaExceededError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(reserve_once, range(8)))
    assert outcomes.count(True) == 2
    assert AlphaVantageQuota("shared-key", path=path).status().hour_used == 2


def test_errors_are_charged_and_secrets_are_not_leaked(monkeypatch, tmp_path):
    ledger = tmp_path / "quota.sqlite3"

    def failed_request(request, timeout):
        raise OSError(f"request failed: {request.full_url}")

    monkeypatch.setattr(av_module, "urlopen", failed_request)
    with pytest.raises(AlphaVantageError, match="request failed") as exc:
        fetch_alpha_vantage(api_key="secret-key", quota_path=ledger, daily_limit=2)
    assert "secret-key" not in str(exc.value)
    assert AlphaVantageQuota("secret-key", path=ledger).status().day_used == 1

    monkeypatch.setattr(
        av_module, "urlopen",
        lambda request, timeout: BytesIO(json.dumps({"Information": "limit for secret-key"}).encode()),
    )
    with pytest.raises(AlphaVantageError, match=r"\[redacted\]"):
        fetch_alpha_vantage(api_key="secret-key", quota_path=ledger, daily_limit=2)
    with pytest.raises(QuotaExceededError):
        fetch_alpha_vantage(api_key="secret-key", quota_path=ledger, daily_limit=2)
    assert AlphaVantageQuota("secret-key", path=ledger).status().day_used == 2


def test_invalid_arguments_do_not_consume_quota(tmp_path):
    ledger = tmp_path / "quota.sqlite3"
    with pytest.raises(ValueError, match="time_from"):
        fetch_alpha_vantage(
            api_key="key", time_from="2026-09-18 10:00", quota_path=ledger
        )
    with pytest.raises(ValueError, match="limit"):
        fetch_alpha_vantage(api_key="key", limit=1001, quota_path=ledger)
    with pytest.raises(ValueError, match="time_from must not be after"):
        fetch_alpha_vantage(
            api_key="key", time_from="20260919T0000", time_to="20260918T0000",
            quota_path=ledger,
        )
    assert not ledger.exists()


def test_empty_feed_has_normalized_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(
        av_module, "urlopen",
        lambda request, timeout: BytesIO(b'{"items": 0, "feed": []}'),
    )
    news = fetch_alpha_vantage(api_key="key", quota_path=tmp_path / "quota.sqlite3")
    assert news.empty
    assert {"news_id", "published_at", "available_at", "ticker", "text"} <= set(news.columns)


@pytest.mark.parametrize("error_type", [AlphaVantageError, QuotaExceededError, RuntimeError])
def test_live_smoke_failure_does_not_show_api_key(monkeypatch, error_type):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-secret")

    def fail_request(**kwargs):
        raise error_type("request failed")

    monkeypatch.setitem(globals(), "fetch_alpha_vantage", fail_request)
    with pytest.raises(pytest.fail.Exception) as error:
        test_live_alpha_vantage_response()
    assert "test-secret" not in str(error.value)
    assert error.value.pytrace is False


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_ALPHA_VANTAGE_SMOKE") != "1",
    reason="opt-in real Alpha Vantage request",
)
def test_live_alpha_vantage_response():
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        pytest.skip("set ALPHAVANTAGE_API_KEY before the live smoke test")
    try:
        news = fetch_alpha_vantage(api_key=key, tickers="AAPL", limit=1)
    except Exception as exc:
        # Pytest's full traceback displays function arguments, including api_key.
        pytest.fail(f"live Alpha Vantage request failed: {type(exc).__name__}", pytrace=False)
    finally:
        del key
    assert {"news_id", "published_at", "available_at", "ticker", "text"} <= set(news.columns)
    assert str(news["available_at"].dtype).endswith("UTC]")
