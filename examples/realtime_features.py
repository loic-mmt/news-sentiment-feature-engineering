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
    build_sentiment_features,
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
    args = parser.parse_args()

    aliases = json.loads(args.aliases.read_text(encoding="utf-8"))
    decision_points = pd.read_csv(args.decisions)
    fetched = fetch_rss(args.feed, ticker_aliases=aliases)
    stored = save_news(fetched, args.news_store)
    scored = attach_sentiment(stored, SentimentAnalyzer())
    features = build_sentiment_features(scored, decision_points)
    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.features_out, engine="pyarrow", index=False)
    print(f"Saved {len(features)} feature rows to {args.features_out}")


if __name__ == "__main__":
    main()
