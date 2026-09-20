"""Offline RSS and Parquet ingestion tests."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

import news_sentiment.news as news_module
from news_sentiment import fetch_rss, save_news

RSS = b"""<?xml version='1.0' encoding='utf-8'?>
<rss version='2.0'><channel><title>Market Wire</title>
<item><title>Apple and Microsoft rally</title>
<description>&lt;p&gt;Shares gain after results.&lt;/p&gt;</description>
<link>https://example.test/one#section</link>
<pubDate>Fri, 18 Sep 2026 08:00:00 GMT</pubDate></item>
<item><title>Markets steady</title><description>No company named.</description>
<link>https://example.test/two</link></item>
</channel></rss>"""

ATOM = b"""<?xml version='1.0' encoding='utf-8'?>
<feed xmlns='http://www.w3.org/2005/Atom'><title>Another Wire</title>
<entry><title>Apple update</title><link href='https://example.test/three'/>
<updated>2026-09-18T09:00:00Z</updated><summary>Apple reports revenue.</summary></entry>
</feed>"""


def test_fetch_rss_atom_aliases_and_unassigned(monkeypatch):
    pytest.importorskip("feedparser")
    payloads = {"https://example.test/rss": RSS, "https://example.test/atom": ATOM}
    monkeypatch.setattr(
        news_module,
        "urlopen",
        lambda request, timeout: BytesIO(payloads[request.full_url]),
    )
    observed = pd.Timestamp("2026-09-18T10:00:00Z")
    fetched = fetch_rss(
        payloads,
        ticker_aliases={"AAPL": ["Apple"], "MSFT": ["Microsoft"]},
        fetched_at=observed,
    )
    assert len(fetched) == 4
    assert set(fetched["ticker"].dropna()) == {"AAPL", "MSFT"}
    assert fetched["ticker"].isna().sum() == 1
    assert fetched["available_at"].eq(observed).all()
    assert fetched["availability_kind"].eq("pipeline_observed").all()
    assert set(fetched["availability_reference"]) == set(payloads)
    assert fetched.loc[0, "published_at"] == pd.Timestamp("2026-09-18T08:00:00Z")
    assert fetched.loc[0, "url"] == "https://example.test/one"
    assert "<p>" not in fetched.loc[0, "text"]


def test_one_failed_feed_warns_but_other_feed_survives(monkeypatch):
    pytest.importorskip("feedparser")

    def open_feed(request, timeout):
        if request.full_url.endswith("bad"):
            raise OSError("unavailable")
        return BytesIO(RSS)

    monkeypatch.setattr(news_module, "urlopen", open_feed)
    with pytest.warns(RuntimeWarning, match="RSS feed failed"):
        result = fetch_rss(["https://example.test/bad", "https://example.test/good"])
    assert len(result) == 2
    with pytest.warns(RuntimeWarning):
        with pytest.raises(RuntimeError, match="all RSS feeds failed"):
            fetch_rss(["https://example.test/bad"])


def test_save_news_is_idempotent_and_preserves_first_availability(tmp_path):
    pytest.importorskip("pyarrow")
    first = pd.DataFrame(
        [{
            "news_id": "one", "published_at": pd.Timestamp("2026-09-18T08:00:00Z"),
            "available_at": pd.Timestamp("2026-09-18T10:00:00Z"),
            "source": "wire", "url": "https://example.test/one", "ticker": "AAPL",
            "title": "Apple rises", "text": "Apple rises",
        }]
    )
    original = first.copy(deep=True)
    path = tmp_path / "news.parquet"
    assert len(save_news(first, path)) == 1
    later = first.copy()
    later["available_at"] = pd.Timestamp("2026-09-18T12:00:00Z")
    stored = save_news(later, path)
    assert len(stored) == 1
    assert stored.loc[0, "available_at"] == pd.Timestamp("2026-09-18T10:00:00Z")
    second_ticker = first.copy()
    second_ticker["ticker"] = "MSFT"
    stored = save_news(second_ticker, path)
    assert len(stored) == 2
    assert set(stored["ticker"]) == {"AAPL", "MSFT"}
    unmatched = first.copy()
    unmatched["ticker"] = None
    stored = save_news(unmatched, path)
    assert len(stored) == 3
    assert len(save_news(unmatched, path)) == 3
    pd.testing.assert_frame_equal(first, original)


def test_fetch_rejects_naive_observation_time():
    pytest.importorskip("feedparser")
    with pytest.raises(ValueError, match="timezone-aware"):
        fetch_rss("https://example.test/rss", fetched_at="2026-09-18 10:00:00")
