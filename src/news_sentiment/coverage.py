"""Separate, timestamped evidence of source coverage."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from .news import _utc_timestamp


COVERAGE_COLUMNS = (
    "source", "ticker", "start_at", "end_at", "recorded_at", "status",
    "evidence_reference",
)


def _empty_coverage() -> pd.DataFrame:
    frame = pd.DataFrame(columns=COVERAGE_COLUMNS)
    for column in ("start_at", "end_at", "recorded_at"):
        frame[column] = pd.Series(dtype="datetime64[ns, UTC]")
    return frame


def validate_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    """Validate a coverage journal; intervals are half-open [start_at, end_at)."""
    if not isinstance(coverage, pd.DataFrame):
        raise TypeError("coverage must be a pandas DataFrame")
    if coverage.columns.has_duplicates:
        raise ValueError("coverage must not have duplicate column names")
    missing = [column for column in COVERAGE_COLUMNS if column not in coverage.columns]
    if missing:
        raise ValueError(f"missing coverage columns: {missing}")
    checked = coverage.copy()
    if checked.empty:
        for column in ("start_at", "end_at", "recorded_at"):
            checked[column] = pd.to_datetime(checked[column], utc=True)
        return checked
    for position, row in enumerate(checked[list(COVERAGE_COLUMNS)].itertuples(index=False)):
        source, ticker, start, end, recorded, status, reference = row
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"coverage row {position}: source must be non-empty")
        if not pd.isna(ticker) and (not isinstance(ticker, str) or not ticker.strip()):
            raise ValueError(f"coverage row {position}: ticker must be a string or missing")
        if status not in {"covered", "incomplete"}:
            raise ValueError(f"coverage row {position}: status must be covered or incomplete")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"coverage row {position}: evidence_reference must be non-empty")
        start = _utc_timestamp(start, field=f"coverage row {position} start_at")
        end = _utc_timestamp(end, field=f"coverage row {position} end_at")
        recorded = _utc_timestamp(recorded, field=f"coverage row {position} recorded_at")
        if start >= end:
            raise ValueError(f"coverage row {position}: start_at must precede end_at")
        for column, value in (("start_at", start), ("end_at", end), ("recorded_at", recorded)):
            checked.iat[position, checked.columns.get_loc(column)] = value
    for column in ("start_at", "end_at", "recorded_at"):
        checked[column] = pd.to_datetime(checked[column], utc=True)
    return checked


def save_coverage(coverage: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Append deduplicated coverage assertions to a separate Parquet journal."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError("Parquet storage requires pip install -e '.[news]'") from exc
    target = Path(path)
    if target.suffix.lower() != ".parquet":
        raise ValueError("path must end in .parquet")
    incoming = validate_coverage(coverage)
    existing = (
        validate_coverage(pd.read_parquet(target, engine="pyarrow"))
        if target.exists() else _empty_coverage()
    )
    combined = pd.concat([existing, incoming], ignore_index=True).drop_duplicates(
        ignore_index=True
    )
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


def coverage_status(
    coverage: pd.DataFrame,
    *,
    ticker: str,
    decision_at: pd.Timestamp,
    lookback: pd.Timedelta,
    required_sources: tuple[str, ...],
    include_at_cutoff: bool,
) -> str:
    """Return covered only when every required source proves the full window."""
    if not required_sources:
        return "unknown"
    start_ns = (decision_at - lookback).value
    end_ns = decision_at.value + int(include_at_cutoff)
    known = coverage[
        coverage["recorded_at"] <= decision_at
        if include_at_cutoff else coverage["recorded_at"] < decision_at
    ]
    overall = "covered"
    for source in required_sources:
        relevant = known[
            (known["source"] == source)
            & (known["ticker"].isna() | (known["ticker"] == ticker))
        ]
        assertions: list[tuple[int, int, int, int, str]] = []
        boundaries = {start_ns, end_ns}
        for row in relevant.itertuples(index=False):
            left = max(start_ns, row.start_at.value)
            right = min(end_ns, row.end_at.value)
            if right <= start_ns or left >= end_ns:
                continue
            boundaries.update((left, right))
            assertions.append((
                left, right, row.recorded_at.value, int(not pd.isna(row.ticker)),
                row.status,
            ))
        ordered = sorted(boundaries)
        for left, right in zip(ordered, ordered[1:]):
            applicable = [
                assertion for assertion in assertions
                if assertion[0] <= left and assertion[1] >= right
            ]
            if not applicable:
                overall = "unknown"
                continue
            latest = max(
                applicable,
                key=lambda assertion: (
                    assertion[2], assertion[3], assertion[4] == "incomplete"
                ),
            )
            if latest[4] == "incomplete":
                return "incomplete"
    return overall
