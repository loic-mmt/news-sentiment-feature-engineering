"""Validate historical articles with explicit availability evidence."""

from __future__ import annotations

import pandas as pd

from .news import AVAILABILITY_COLUMNS, _validated_news


HISTORICAL_AVAILABILITY_KINDS = frozenset({"provider_first_seen", "archive_first_seen"})


def import_historical_news(articles: pd.DataFrame) -> pd.DataFrame:
    """Validate historical news before storing or scoring it.

    ``available_at`` must be a timezone-aware first-availability timestamp,
    supported by a per-row ``availability_kind`` and non-empty
    ``availability_reference`` (for example a provider field or archive snapshot
    identifier). This checks the claim's structure, not the external evidence.
    Today's ingestion time must not be relabeled as historical availability.
    """
    if not isinstance(articles, pd.DataFrame):
        raise TypeError("articles must be a pandas DataFrame")
    if articles.columns.has_duplicates:
        raise ValueError("articles must not have duplicate column names")
    missing = [column for column in AVAILABILITY_COLUMNS if column not in articles.columns]
    if missing:
        raise ValueError(f"missing historical availability columns: {missing}")
    checked = _validated_news(articles)
    if checked.duplicated(subset=["news_id", "ticker"]).any():
        raise ValueError("historical news must have unique (news_id, ticker) pairs")
    for position, row in enumerate(checked.itertuples(index=False)):
        if not isinstance(row.ticker, str) or not row.ticker.strip():
            raise ValueError(f"row {position}: historical ticker must be non-empty")
        if not isinstance(row.source, str) or not row.source.strip():
            raise ValueError(f"row {position}: historical source must be non-empty")
        if pd.isna(row.published_at):
            raise ValueError(f"row {position}: published_at is required")
        if row.available_at < row.published_at:
            raise ValueError(f"row {position}: available_at precedes published_at")
        if row.availability_kind not in HISTORICAL_AVAILABILITY_KINDS:
            raise ValueError(
                f"row {position}: availability_kind must be provider_first_seen "
                "or archive_first_seen"
            )
        if (
            not isinstance(row.availability_reference, str)
            or not row.availability_reference.strip()
        ):
            raise ValueError(f"row {position}: availability_reference must identify evidence")
    return checked
