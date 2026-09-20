"""Local, process-safe accounting for Alpha Vantage API requests."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


DEFAULT_HOURLY_LIMIT = 25  # Local policy; Alpha Vantage does not publish an hourly cap.
DEFAULT_DAILY_LIMIT = 25  # Standard free plan, conservatively enforced over rolling 24h.


class QuotaExceededError(RuntimeError):
    """A local request limit has been reached before contacting the provider."""


@dataclass(frozen=True)
class QuotaStatus:
    hour_used: int
    hour_limit: int | None
    hour_remaining: int | None
    day_used: int
    day_limit: int | None
    day_remaining: int | None


def _limit(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None")
    return value


def _default_path() -> Path:
    configured = os.environ.get("NEWS_SENTIMENT_QUOTA_DB")
    return Path(configured).expanduser() if configured else Path.home() / ".local/share/news_sentiment/alpha_vantage_quota.sqlite3"


class AlphaVantageQuota:
    """Count requests for one API key in rolling 1h and 24h windows.

    The SQLite ledger is shared by processes using the same path. Only a SHA-256
    fingerprint of the key is stored. Reservations are atomic and made *before*
    network I/O, so failed requests still consume a local slot.
    """

    def __init__(
        self,
        api_key: str,
        *,
        path: str | Path | None = None,
        hourly_limit: int | None = DEFAULT_HOURLY_LIMIT,
        daily_limit: int | None = DEFAULT_DAILY_LIMIT,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        self._key_hash = hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()
        self.path = Path(path).expanduser() if path is not None else _default_path()
        self.hourly_limit = _limit(hourly_limit, "hourly_limit")
        self.daily_limit = _limit(daily_limit, "daily_limit")

    def _now_epoch(self) -> float:
        return time.time()

    def _status(self, connection: sqlite3.Connection, now: float) -> QuotaStatus:
        hour_used, day_used = connection.execute(
            """
            SELECT
                COUNT(CASE WHEN called_at > ? THEN 1 END),
                COUNT(*)
            FROM calls
            WHERE key_hash = ? AND called_at > ?
            """,
            (now - 3600, self._key_hash, now - 86400),
        ).fetchone()
        return QuotaStatus(
            hour_used=hour_used,
            hour_limit=self.hourly_limit,
            hour_remaining=(
                None if self.hourly_limit is None else max(0, self.hourly_limit - hour_used)
            ),
            day_used=day_used,
            day_limit=self.daily_limit,
            day_remaining=(
                None if self.daily_limit is None else max(0, self.daily_limit - day_used)
            ),
        )

    def status(self) -> QuotaStatus:
        """Return local usage without making an API request or creating a ledger."""
        if not self.path.exists():
            return QuotaStatus(
                0, self.hourly_limit, self.hourly_limit,
                0, self.daily_limit, self.daily_limit,
            )
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            return self._status(connection, self._now_epoch())

    def reserve(self) -> QuotaStatus:
        """Atomically charge one attempted call, or refuse before network I/O."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS calls (key_hash TEXT NOT NULL, called_at REAL NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS calls_key_time ON calls (key_hash, called_at)"
            )
            now = self._now_epoch()
            before = self._status(connection, now)
            if before.hour_remaining == 0 or before.day_remaining == 0:
                raise QuotaExceededError(
                    "local Alpha Vantage quota reached "
                    f"(1h: {before.hour_used}/{before.hour_limit or 'unlimited'}, "
                    f"24h: {before.day_used}/{before.day_limit or 'unlimited'})"
                )
            connection.execute(
                "INSERT INTO calls (key_hash, called_at) VALUES (?, ?)",
                (self._key_hash, now),
            )
            return self._status(connection, now)
