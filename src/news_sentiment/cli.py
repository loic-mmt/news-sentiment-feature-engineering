"""Small command-line entry point for locally tracked provider quotas."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from .quota import AlphaVantageQuota, DEFAULT_DAILY_LIMIT, DEFAULT_HOURLY_LIMIT


def _parse_limit(value: str) -> int | None:
    if value.lower() in {"none", "unlimited"}:
        return None
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit must be positive or 'unlimited'") from exc
    if limit <= 0:
        raise argparse.ArgumentTypeError("limit must be positive or 'unlimited'")
    return limit


def _line(label: str, used: int, limit: int | None, remaining: int | None) -> str:
    if limit is None:
        return f"{label}: {used} appels ; plafond local désactivé"
    return f"{label}: {used}/{limit} appels ; reste {remaining}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="news-sentiment")
    providers = parser.add_subparsers(dest="provider", required=True)
    alpha = providers.add_parser("alpha-vantage", help="Alpha Vantage tools")
    actions = alpha.add_subparsers(dest="action", required=True)
    quota = actions.add_parser("quota", help="show local usage and remaining calls")
    quota.add_argument("--quota-db", help="SQLite ledger path")
    quota.add_argument("--hourly-limit", type=_parse_limit, default=DEFAULT_HOURLY_LIMIT)
    quota.add_argument("--daily-limit", type=_parse_limit, default=DEFAULT_DAILY_LIMIT)
    args = parser.parse_args(argv)

    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        print("ALPHAVANTAGE_API_KEY must be set", file=sys.stderr)
        return 2
    tracker = AlphaVantageQuota(
        key,
        path=args.quota_db,
        hourly_limit=args.hourly_limit,
        daily_limit=args.daily_limit,
    )
    status = tracker.status()
    print(_line("1h", status.hour_used, status.hour_limit, status.hour_remaining))
    print(_line("24h", status.day_used, status.day_limit, status.day_remaining))
    print("Compteur local uniquement ; appels faits ailleurs non inclus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
