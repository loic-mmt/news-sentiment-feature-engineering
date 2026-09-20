"""Collect configured feeds and export FinBERT features at supplied decision times.

Example:
    python examples/realtime_features.py \
      --feed https://example.com/finance.xml \
      --aliases aliases.json --decisions decisions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from news_sentiment import (
    SentimentAnalyzer,
    attach_sentiment,
    export_sentiment_features,
    fetch_rss,
    save_news,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed", action="append", required=True, help="RSS/Atom URL; repeatable")
    parser.add_argument("--aliases", type=Path, required=True, help="JSON ticker -> alias list")
    parser.add_argument("--decisions", type=Path, required=True, help="CSV with ticker,decision_at")
    parser.add_argument("--news-store", type=Path, default=Path("data/news.parquet"))
    parser.add_argument("--features-out", type=Path, default=Path("data/features.parquet"))
    parser.add_argument("--coverage", type=Path, help="Separate coverage journal in Parquet")
    parser.add_argument("--required-source", action="append", help="Source identifier; repeatable")
    parser.add_argument("--exclude-cutoff", action="store_true", help="Require available_at < decision_at")
    args = parser.parse_args()
    if bool(args.coverage) != bool(args.required_source):
        parser.error("--coverage and at least one --required-source must be provided together")

    aliases = json.loads(args.aliases.read_text(encoding="utf-8"))
    decision_points = pd.read_csv(args.decisions)
    fetched = fetch_rss(args.feed, ticker_aliases=aliases)
    stored = save_news(fetched, args.news_store)
    analyzer = SentimentAnalyzer()
    scored = attach_sentiment(stored, analyzer)
    coverage = pd.read_parquet(args.coverage) if args.coverage else None
    features = export_sentiment_features(
        scored, decision_points, args.features_out,
        checkpoint=analyzer.model_name,
        input_identifiers={"news_store": args.news_store.name, "decisions": args.decisions.name},
        coverage=coverage, required_sources=args.required_source,
        include_at_cutoff=not args.exclude_cutoff,
    )
    print(f"Saved {len(features)} feature rows to {args.features_out} and its manifest")


if __name__ == "__main__":
    main()
