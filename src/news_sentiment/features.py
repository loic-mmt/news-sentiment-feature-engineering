"""Point-in-time sentiment features from scored, timestamped news."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .coverage import _empty_coverage, coverage_status, validate_coverage
from .results import PREDICTION_COLUMNS
from .stats import validate_prediction_frame

if TYPE_CHECKING:
    from .analyzer import SentimentAnalyzer

SCORE_COLUMNS = tuple(column for column in PREDICTION_COLUMNS if column != "text")
FEATURE_COLUMNS = (
    "ticker",
    "decision_at",
    "news_count",
    "coverage_status",
    "last_contributing_available_at",
    "sentiment_mean",
    "sentiment_std",
    "p_positive_mean",
    "p_neutral_mean",
    "p_negative_mean",
    "positive_share",
    "negative_share",
    "confidence_mean",
    "sentiment_ewm",
    "sentiment_momentum",
    "hours_since_last_news",
)


def _utc_timestamp(value: object, *, field: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a timezone-aware timestamp") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    return timestamp.tz_convert("UTC")


def _duration(value: str | pd.Timedelta, *, field: str) -> pd.Timedelta:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a positive duration")
    try:
        duration = pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive duration") from exc
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        raise ValueError(f"{field} must be a positive duration")
    return duration


def attach_sentiment(
    news: pd.DataFrame,
    analyzer: SentimentAnalyzer,
    *,
    text_col: str = "text",
) -> pd.DataFrame:
    """Attach a single inference batch to news without changing its rows or index."""
    if not isinstance(news, pd.DataFrame):
        raise TypeError("news must be a pandas DataFrame")
    if not isinstance(text_col, str) or not text_col.strip():
        raise ValueError("text_col must be a non-empty column name")
    if text_col not in news.columns:
        raise ValueError(f"news is missing text column {text_col!r}")
    if not callable(getattr(analyzer, "predict", None)):
        raise TypeError("analyzer must provide a predict(texts) method")
    if any(column in news.columns for column in SCORE_COLUMNS):
        raise ValueError("news already contains sentiment prediction columns")

    output = news.copy()
    texts = output[text_col].tolist()
    for position, value in enumerate(texts):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"row {position}: {text_col} must be non-empty text")

    if not texts:
        if "text" not in output.columns:
            output["text"] = pd.Series(dtype="object")
        for column in SCORE_COLUMNS:
            output[column] = pd.Series(dtype="object" if column == "label" else "float64")
        return output

    predictions = analyzer.predict(texts)
    if not isinstance(predictions, pd.DataFrame) or len(predictions) != len(output):
        raise ValueError("analyzer must return one prediction per news row")
    validate_prediction_frame(predictions)
    if predictions["text"].tolist() != texts:
        raise ValueError("analyzer predictions are not aligned with news text order")

    if "text" not in output.columns:
        output["text"] = predictions["text"].to_numpy()
    for column in SCORE_COLUMNS:
        output[column] = predictions[column].to_numpy()
    return output


def build_sentiment_features(
    scored_news: pd.DataFrame,
    decision_points: pd.DataFrame,
    *,
    lookback: str | pd.Timedelta = "24h",
    short_lookback: str | pd.Timedelta = "6h",
    half_life: str | pd.Timedelta = "6h",
    include_at_cutoff: bool = True,
    coverage: pd.DataFrame | None = None,
    required_sources: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Calculate one backward-looking feature row per (ticker, decision_at)."""
    if not isinstance(scored_news, pd.DataFrame):
        raise TypeError("scored_news must be a pandas DataFrame")
    if not isinstance(decision_points, pd.DataFrame):
        raise TypeError("decision_points must be a pandas DataFrame")
    required_news = ("news_id", "ticker", "available_at", *PREDICTION_COLUMNS)
    missing_news = [column for column in required_news if column not in scored_news.columns]
    if missing_news:
        raise ValueError(f"missing scored_news columns: {missing_news}")
    missing_points = [
        column for column in ("ticker", "decision_at")
        if column not in decision_points.columns
    ]
    if missing_points:
        raise ValueError(f"missing decision_points columns: {missing_points}")
    if scored_news.columns.has_duplicates or decision_points.columns.has_duplicates:
        raise ValueError("input DataFrames must not have duplicate column names")
    if not isinstance(include_at_cutoff, bool):
        raise TypeError("include_at_cutoff must be a boolean")
    if required_sources is None:
        sources: tuple[str, ...] = ()
    else:
        if isinstance(required_sources, (str, bytes)):
            raise TypeError("required_sources must be an iterable of source identifiers")
        sources = tuple(required_sources)
        if not sources or any(
            not isinstance(source, str) or not source.strip() for source in sources
        ):
            raise ValueError("required_sources must contain non-empty source identifiers")
        if len(set(sources)) != len(sources):
            raise ValueError("required_sources must not contain duplicates")
    checked_coverage = validate_coverage(coverage) if coverage is not None else _empty_coverage()
    if not checked_coverage.empty and not sources:
        raise ValueError("required_sources is needed to interpret a coverage journal")

    long_window = _duration(lookback, field="lookback")
    short_window = _duration(short_lookback, field="short_lookback")
    decay_half_life = _duration(half_life, field="half_life")
    if short_window >= long_window:
        raise ValueError("short_lookback must be shorter than lookback")

    points: list[tuple[str, pd.Timestamp]] = []
    seen_points: set[tuple[str, pd.Timestamp]] = set()
    for position, (ticker, value) in enumerate(
        decision_points[["ticker", "decision_at"]].itertuples(index=False, name=None)
    ):
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError(f"decision_points row {position}: ticker must be non-empty")
        point = (ticker, _utc_timestamp(value, field=f"decision_points row {position} decision_at"))
        if point in seen_points:
            raise ValueError(f"duplicate decision point: {point}")
        seen_points.add(point)
        points.append(point)

    if not scored_news.empty:
        validate_prediction_frame(scored_news)

    prepared = scored_news.copy()
    timestamps: list[pd.Timestamp] = []
    for position, (news_id, ticker, value) in enumerate(
        prepared[["news_id", "ticker", "available_at"]].itertuples(index=False, name=None)
    ):
        if not isinstance(news_id, str) or not news_id.strip():
            raise ValueError(f"scored_news row {position}: news_id must be non-empty")
        if not pd.isna(ticker) and (not isinstance(ticker, str) or not ticker.strip()):
            raise ValueError(f"scored_news row {position}: ticker must be a string or missing")
        timestamps.append(_utc_timestamp(value, field=f"scored_news row {position} available_at"))
    prepared["available_at"] = pd.to_datetime(timestamps, utc=True)
    prepared = prepared.dropna(subset=["ticker"])
    prepared = prepared.sort_values("available_at", kind="stable")
    prepared = prepared.drop_duplicates(subset=["ticker", "news_id"], keep="first")

    groups: dict[str, tuple[pd.DataFrame, list[int]]] = {}
    for ticker, group in prepared.groupby("ticker", sort=False):
        ordered = group.reset_index(drop=True)
        groups[ticker] = (ordered, [timestamp.value for timestamp in ordered["available_at"]])

    records: list[dict[str, object]] = []
    for ticker, decision_at in points:
        record: dict[str, object] = {column: math.nan for column in FEATURE_COLUMNS}
        record.update(
            ticker=ticker, decision_at=decision_at, news_count=0,
            coverage_status=coverage_status(
                checked_coverage, ticker=ticker, decision_at=decision_at,
                lookback=long_window, required_sources=sources,
                include_at_cutoff=include_at_cutoff,
            ),
            last_contributing_available_at=pd.NaT,
        )
        if ticker not in groups:
            records.append(record)
            continue

        group, times = groups[ticker]
        end = (bisect_right if include_at_cutoff else bisect_left)(times, decision_at.value)
        if end:
            record["hours_since_last_news"] = (
                decision_at - group.iloc[end - 1]["available_at"]
            ).total_seconds() / 3600.0

        start = bisect_right(times, (decision_at - long_window).value)
        if start == end:
            records.append(record)
            continue

        window = group.iloc[start:end]
        score = window["sentiment_score"].to_numpy(dtype=float)
        confidence = window["confidence"].to_numpy(dtype=float)
        ages_ns = (
            (decision_at - window["available_at"])
            .to_numpy(dtype="timedelta64[ns]")
            .astype(np.int64)
        )
        # Removing the common minimum age leaves the weighted mean unchanged,
        # while preventing every weight from underflowing for tiny half-lives.
        weights = confidence * np.exp(
            -math.log(2) * (ages_ns - ages_ns.min()) / decay_half_life.value
        )
        short_start = bisect_right(times, (decision_at - short_window).value)
        short_scores = group.iloc[short_start:end]["sentiment_score"].to_numpy(dtype=float)

        record.update(
            news_count=int(len(window)),
            last_contributing_available_at=window.iloc[-1]["available_at"],
            sentiment_mean=float(np.mean(score)),
            sentiment_std=float(np.std(score, ddof=0)),
            p_positive_mean=float(window["p_positive"].mean()),
            p_neutral_mean=float(window["p_neutral"].mean()),
            p_negative_mean=float(window["p_negative"].mean()),
            positive_share=float((window["label"] == "positive").mean()),
            negative_share=float((window["label"] == "negative").mean()),
            confidence_mean=float(np.mean(confidence)),
            sentiment_ewm=float(np.average(score, weights=weights)),
            sentiment_momentum=(
                float(np.mean(short_scores) - np.mean(score))
                if len(short_scores)
                else math.nan
            ),
        )
        records.append(record)

    return pd.DataFrame.from_records(records, columns=FEATURE_COLUMNS)
