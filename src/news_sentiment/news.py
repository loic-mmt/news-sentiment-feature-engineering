"""Collect and persist timestamped financial news from configurable feeds."""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
import warnings
from calendar import timegm
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import pandas as pd

NEWS_COLUMNS = (
    "news_id",
    "published_at",
    "available_at",
    "source",
    "url",
    "ticker",
    "title",
    "text",
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def _plain_text(value: object) -> str:
    parser = _TextExtractor()
    parser.feed(str(value or ""))
    return " ".join(" ".join(parser.parts).split())


def _utc_timestamp(value: object, *, field: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a timezone-aware timestamp") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    return timestamp.tz_convert("UTC")


def _canonical_url(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, "")
    )


def _alias_patterns(
    ticker_aliases: Mapping[str, Iterable[str]] | None,
) -> dict[str, list[re.Pattern[str]]]:
    if ticker_aliases is None:
        return {}
    if not isinstance(ticker_aliases, Mapping):
        raise TypeError("ticker_aliases must map tickers to their aliases")

    patterns: dict[str, list[re.Pattern[str]]] = {}
    for ticker, aliases in ticker_aliases.items():
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("ticker_aliases keys must be non-empty ticker strings")
        if isinstance(aliases, str):
            aliases = [aliases]
        try:
            alias_list = list(aliases)
        except TypeError as exc:
            raise TypeError(f"aliases for {ticker!r} must be iterable") from exc
        if not alias_list or any(not isinstance(a, str) or not a.strip() for a in alias_list):
            raise ValueError(f"aliases for {ticker!r} must contain non-empty strings")
        patterns[ticker.strip()] = [
            re.compile(rf"(?<!\w){re.escape(alias.strip())}(?!\w)", re.IGNORECASE)
            for alias in alias_list
        ]
    return patterns


def _empty_news() -> pd.DataFrame:
    frame = pd.DataFrame(columns=NEWS_COLUMNS)
    for column in ("published_at", "available_at"):
        frame[column] = pd.Series(dtype="datetime64[ns, UTC]")
    return frame


def fetch_rss(
    feed_urls: Iterable[str] | str,
    *,
    ticker_aliases: Mapping[str, Iterable[str]] | None = None,
    fetched_at: datetime | pd.Timestamp | None = None,
    timeout_seconds: float = 10,
) -> pd.DataFrame:
    """Fetch RSS/Atom titles and summaries, using first observation as availability.

    An article matching several aliases gets one row per ticker. Unmatched articles
    retain a missing ticker so they can still be stored and inspected.
    """
    try:
        import feedparser
    except ImportError as exc:
        raise ImportError("RSS collection requires pip install -e '.[news]'") from exc

    urls = [feed_urls] if isinstance(feed_urls, str) else list(feed_urls)
    if not urls:
        raise ValueError("feed_urls must contain at least one URL")
    for url in urls:
        if not isinstance(url, str) or not _canonical_url(url):
            raise ValueError(f"invalid HTTP(S) feed URL: {url!r}")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise TypeError("timeout_seconds must be a positive real number")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive finite number")

    fixed_observation = (
        _utc_timestamp(fetched_at, field="fetched_at")
        if fetched_at is not None
        else None
    )
    patterns = _alias_patterns(ticker_aliases)
    records: list[dict[str, object]] = []
    successful_feeds = 0
    failures: list[str] = []

    for feed_url in urls:
        try:
            request = Request(feed_url, headers={"User-Agent": "news-sentiment/0.1"})
            with urlopen(request, timeout=timeout_seconds) as response:
                parsed = feedparser.parse(response.read())
            if parsed.get("bozo") and not parsed.entries:
                raise ValueError(f"invalid feed: {parsed.get('bozo_exception')}")
        except (OSError, ValueError, TypeError) as exc:
            failures.append(f"{feed_url}: {exc}")
            warnings.warn(f"RSS feed failed: {feed_url}: {exc}", RuntimeWarning, stacklevel=2)
            continue

        successful_feeds += 1
        # Mark availability only after the feed has been received and parsed.
        # Using the batch start time could expose a slow response too early.
        observed_at = fixed_observation or pd.Timestamp.now(tz="UTC")
        source = _plain_text(parsed.feed.get("title")) or feed_url
        for entry in parsed.entries:
            title = _plain_text(entry.get("title"))
            summary = _plain_text(entry.get("summary") or entry.get("description"))
            if not summary and entry.get("content"):
                summary = _plain_text(entry.content[0].get("value"))
            if not title and not summary:
                continue
            text = (
                f"{title}. {summary}"
                if title and summary and title != summary
                else title or summary
            )
            url = _canonical_url(entry.get("link"))
            identity = f"url:{url}" if url else f"content:{title.casefold()}\n{text.casefold()}"
            news_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            parsed_date = entry.get("published_parsed") or entry.get("updated_parsed")
            try:
                published_at = (
                    pd.Timestamp(datetime.fromtimestamp(timegm(parsed_date), tz=timezone.utc))
                    if parsed_date is not None
                    else pd.NaT
                )
            except (OverflowError, ValueError, TypeError):
                published_at = pd.NaT
            haystack = f"{title} {summary}"
            tickers = [
                ticker
                for ticker, aliases in patterns.items()
                if any(pattern.search(haystack) for pattern in aliases)
            ] or [None]
            for ticker in tickers:
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
                    }
                )

    if successful_feeds == 0:
        raise RuntimeError("all RSS feeds failed: " + "; ".join(failures))
    if not records:
        return _empty_news()
    news = pd.DataFrame.from_records(records, columns=NEWS_COLUMNS)
    return news.drop_duplicates(subset=["news_id", "ticker"], keep="first").reset_index(drop=True)


def _validated_news(news: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(news, pd.DataFrame):
        raise TypeError("news must be a pandas DataFrame")
    missing = [column for column in NEWS_COLUMNS if column not in news.columns]
    if missing:
        raise ValueError(f"missing news columns: {missing}")
    checked = news.copy()
    if checked.empty:
        return checked
    for position, row in enumerate(checked[list(NEWS_COLUMNS)].itertuples(index=False, name=None)):
        news_id, published_at, available_at, source, url, ticker, title, text = row
        if not isinstance(news_id, str) or not news_id.strip():
            raise ValueError(f"row {position}: news_id must be a non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"row {position}: text must be a non-empty string")
        if not pd.isna(ticker) and (not isinstance(ticker, str) or not ticker.strip()):
            raise ValueError(f"row {position}: ticker must be a string or missing")
        checked.iat[position, checked.columns.get_loc("available_at")] = _utc_timestamp(
            available_at, field=f"row {position} available_at"
        )
        if not pd.isna(published_at):
            checked.iat[position, checked.columns.get_loc("published_at")] = _utc_timestamp(
                published_at, field=f"row {position} published_at"
            )
    checked["available_at"] = pd.to_datetime(checked["available_at"], utc=True)
    checked["published_at"] = pd.to_datetime(checked["published_at"], utc=True)
    return checked


def save_news(news: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Atomically merge news into a local Parquet file and return all stored rows.

    Re-observing an article/ticker pair never moves its availability time later.
    This operation assumes one writer per file.
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError("Parquet storage requires pip install -e '.[news]'") from exc

    target = Path(path)
    if target.suffix.lower() != ".parquet":
        raise ValueError("path must end in .parquet")
    incoming = _validated_news(news)
    existing = (
        _validated_news(pd.read_parquet(target, engine="pyarrow"))
        if target.exists()
        else _empty_news()
    )
    combined = pd.concat([existing, incoming], ignore_index=True)
    if not combined.empty:
        combined = combined.sort_values("available_at", kind="stable")
        combined = combined.drop_duplicates(subset=["news_id", "ticker"], keep="first")
        combined = combined.reset_index(drop=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}-", suffix=".parquet", dir=target.parent
    )
    os.close(descriptor)
    try:
        combined.to_parquet(temporary_name, engine="pyarrow", index=False)
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return combined
