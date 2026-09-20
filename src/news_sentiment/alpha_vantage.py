"""Alpha Vantage NEWS_SENTIMENT connector, normalized for point-in-time features."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .news import NEWS_COLUMNS, _canonical_url, _plain_text, _utc_timestamp
from .quota import AlphaVantageQuota, DEFAULT_DAILY_LIMIT, DEFAULT_HOURLY_LIMIT


ALPHA_VANTAGE_COLUMNS = (
    *NEWS_COLUMNS,
    "av_relevance_score",
    "av_ticker_sentiment_score",
    "av_ticker_sentiment_label",
    "av_overall_sentiment_score",
    "av_overall_sentiment_label",
)


class AlphaVantageError(RuntimeError):
    """The provider request or its response could not be used."""


def _csv_filter(values: str | Iterable[str] | None, name: str) -> str | None:
    if values is None:
        return None
    items = values.split(",") if isinstance(values, str) else list(values)
    if not items or any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError(f"{name} must contain non-empty strings")
    return ",".join(item.strip() for item in items)


def _api_time(value: object | None, name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and re.fullmatch(r"\d{8}T\d{4}", value):
        try:
            datetime.strptime(value, "%Y%m%dT%H%M")
        except ValueError as exc:
            raise ValueError(f"{name} must be a valid UTC datetime") from exc
        return value
    return _utc_timestamp(value, field=name).strftime("%Y%m%dT%H%M")


def _published_at(value: object) -> pd.Timestamp | pd.NaT:
    if not isinstance(value, str):
        return pd.NaT
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            return pd.Timestamp(datetime.strptime(value, fmt).replace(tzinfo=timezone.utc))
        except ValueError:
            continue
    return pd.NaT


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _empty_news() -> pd.DataFrame:
    frame = pd.DataFrame(columns=ALPHA_VANTAGE_COLUMNS)
    for column in ("published_at", "available_at"):
        frame[column] = pd.Series(dtype="datetime64[ns, UTC]")
    return frame


def fetch_alpha_vantage(
    *,
    api_key: str | None = None,
    tickers: str | Iterable[str] | None = None,
    topics: str | Iterable[str] | None = None,
    time_from: object | None = None,
    time_to: object | None = None,
    sort: str = "LATEST",
    limit: int = 50,
    fetched_at: object | None = None,
    timeout_seconds: float = 10,
    quota_path: str | Path | None = None,
    hourly_limit: int | None = DEFAULT_HOURLY_LIMIT,
    daily_limit: int | None = DEFAULT_DAILY_LIMIT,
) -> pd.DataFrame:
    """Fetch one NEWS_SENTIMENT page and charge one locally tracked API attempt.

    Multiple tickers in the API filter mean *all* must occur in an article (AND),
    not one independent query per ticker. All provider ticker tags are retained.
    ``available_at`` is the response observation time, never publication time.
    """
    key = api_key if api_key is not None else os.environ.get("ALPHAVANTAGE_API_KEY")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("set ALPHAVANTAGE_API_KEY or pass api_key")
    key = key.strip()
    ticker_filter = _csv_filter(tickers, "tickers")
    topic_filter = _csv_filter(topics, "topics")
    start = _api_time(time_from, "time_from")
    end = _api_time(time_to, "time_to")
    if start is not None and end is not None and start > end:
        raise ValueError("time_from must not be after time_to")
    if not isinstance(sort, str) or sort.upper() not in {"LATEST", "EARLIEST", "RELEVANCE"}:
        raise ValueError("sort must be LATEST, EARLIEST, or RELEVANCE")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer between 1 and 1000")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive finite number")
    fixed_observation = (
        _utc_timestamp(fetched_at, field="fetched_at") if fetched_at is not None else None
    )

    quota = AlphaVantageQuota(
        key, path=quota_path, hourly_limit=hourly_limit, daily_limit=daily_limit
    )
    parameters = {"function": "NEWS_SENTIMENT", "sort": sort.upper(), "limit": limit}
    for name, value in (
        ("tickers", ticker_filter), ("topics", topic_filter),
        ("time_from", start), ("time_to", end),
    ):
        if value is not None:
            parameters[name] = value
    parameters["apikey"] = key
    request = Request(
        "https://www.alphavantage.co/query?" + urlencode(parameters),
        headers={"User-Agent": "news-sentiment/0.1", "Accept": "application/json"},
    )

    quota.reserve()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except OSError:
        # HTTPError and URLError can contain the URL (including the API key).
        raise AlphaVantageError("Alpha Vantage request failed") from None
    except (ValueError, UnicodeError):
        raise AlphaVantageError("Alpha Vantage returned invalid JSON") from None

    if not isinstance(payload, Mapping):
        raise AlphaVantageError("Alpha Vantage returned an unexpected response")
    for field in ("Error Message", "Information", "Note"):
        if field in payload:
            message = str(payload[field]).replace(key, "[redacted]")
            raise AlphaVantageError(f"Alpha Vantage {field}: {message}")
    feed = payload.get("feed")
    if not isinstance(feed, list):
        raise AlphaVantageError("Alpha Vantage response has no news feed")

    observed_at = fixed_observation or pd.Timestamp.now(tz="UTC")
    records: list[dict[str, object]] = []
    for entry in feed:
        if not isinstance(entry, Mapping):
            continue
        title = _plain_text(entry.get("title"))
        summary = _plain_text(entry.get("summary"))
        if not title and not summary:
            continue
        text = f"{title}. {summary}" if title and summary and title != summary else title or summary
        url = _canonical_url(entry.get("url"))
        source = _plain_text(entry.get("source")) or "Alpha Vantage"
        published_at = _published_at(entry.get("time_published"))
        identity = (
            f"url:{url}" if url else
            f"content:{source.casefold()}\n{title.casefold()}\n{text.casefold()}"
        )
        news_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        tags = entry.get("ticker_sentiment")
        tags = tags if isinstance(tags, list) else []
        tagged: dict[str, Mapping[str, object]] = {}
        for tag in tags:
            if not isinstance(tag, Mapping):
                continue
            ticker = tag.get("ticker")
            if isinstance(ticker, str) and ticker.strip():
                tagged.setdefault(ticker.strip(), tag)
        for ticker, tag in (tagged.items() if tagged else [(None, {})]):
            records.append(
                {
                    "news_id": news_id,
                    "published_at": published_at,
                    "available_at": observed_at,
                    "source": source,
                    "url": url,
                    "ticker": ticker,
                    "title": title,
                    "text": text,
                    "av_relevance_score": _number(tag.get("relevance_score")),
                    "av_ticker_sentiment_score": _number(tag.get("ticker_sentiment_score")),
                    "av_ticker_sentiment_label": tag.get("ticker_sentiment_label"),
                    "av_overall_sentiment_score": _number(entry.get("overall_sentiment_score")),
                    "av_overall_sentiment_label": entry.get("overall_sentiment_label"),
                }
            )
    if not records:
        return _empty_news()
    return (
        pd.DataFrame.from_records(records, columns=ALPHA_VANTAGE_COLUMNS)
        .drop_duplicates(subset=["news_id", "ticker"], keep="first")
        .reset_index(drop=True)
    )
