"""Reproducible feature exports with a verifiable sidecar manifest."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .features import _duration, build_sentiment_features
from .history import HISTORICAL_AVAILABILITY_KINDS, import_historical_news
from .news import AVAILABILITY_COLUMNS, _validated_news


FEATURE_SCHEMA_VERSION = "1.0"


def _frame_fingerprint(frame: pd.DataFrame) -> dict[str, object]:
    payload = frame.to_json(orient="split", date_format="iso", date_unit="ns")
    return {"rows": len(frame), "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_export_news(scored_news: pd.DataFrame) -> None:
    checked = _validated_news(scored_news)
    missing = [column for column in AVAILABILITY_COLUMNS if column not in checked.columns]
    if missing:
        raise ValueError(f"export requires availability provenance columns: {missing}")
    for position, row in enumerate(checked.itertuples(index=False)):
        if row.availability_kind not in {*HISTORICAL_AVAILABILITY_KINDS, "pipeline_observed"}:
            raise ValueError(f"row {position}: unknown availability_kind")
        if not isinstance(row.availability_reference, str) or not row.availability_reference.strip():
            raise ValueError(f"row {position}: availability_reference must identify evidence")
        if not isinstance(row.source, str) or not row.source.strip():
            raise ValueError(f"row {position}: source must be non-empty")
    historical = checked[checked["availability_kind"].isin(HISTORICAL_AVAILABILITY_KINDS)]
    if not historical.empty:
        import_historical_news(historical)


def export_sentiment_features(
    scored_news: pd.DataFrame,
    decision_points: pd.DataFrame,
    path: str | Path,
    *,
    checkpoint: str,
    input_identifiers: Mapping[str, str] | None = None,
    coverage: pd.DataFrame | None = None,
    required_sources: Iterable[str] | None = None,
    lookback: str | pd.Timedelta = "24h",
    short_lookback: str | pd.Timedelta = "6h",
    half_life: str | pd.Timedelta = "6h",
    include_at_cutoff: bool = True,
) -> pd.DataFrame:
    """Build features and write Parquet plus ``.manifest.json`` beside it.

    The manifest records schema, windows, checkpoint, caller input identifiers,
    input fingerprints, and the Parquet checksum. A mismatched pair is detectable
    after an interrupted write.
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError("Parquet export requires pip install -e '.[news]'") from exc
    target = Path(path)
    if target.suffix.lower() != ".parquet":
        raise ValueError("path must end in .parquet")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("checkpoint must identify the model used for scoring")
    _validate_export_news(scored_news)
    if input_identifiers is None:
        identifiers: dict[str, str] = {}
    else:
        if not isinstance(input_identifiers, Mapping):
            raise TypeError("input_identifiers must map names to identifiers")
        identifiers = dict(input_identifiers)
        if any(
            not isinstance(name, str) or not name.strip()
            or not isinstance(value, str) or not value.strip()
            for name, value in identifiers.items()
        ):
            raise ValueError("input_identifiers must contain non-empty strings")
    if isinstance(required_sources, (str, bytes)):
        raise TypeError("required_sources must be an iterable of source identifiers")
    sources = tuple(required_sources) if required_sources is not None else None
    features = build_sentiment_features(
        scored_news, decision_points,
        lookback=lookback, short_lookback=short_lookback, half_life=half_life,
        include_at_cutoff=include_at_cutoff, coverage=coverage,
        required_sources=sources,
    )
    manifest = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint.strip(),
        "aggregation": {
            "lookback": str(_duration(lookback, field="lookback")),
            "short_lookback": str(_duration(short_lookback, field="short_lookback")),
            "half_life": str(_duration(half_life, field="half_life")),
            "include_at_cutoff": include_at_cutoff,
            "required_sources": list(sources or ()),
        },
        "input_identifiers": identifiers,
        "inputs": {
            "scored_news": _frame_fingerprint(scored_news),
            "decision_points": _frame_fingerprint(decision_points),
            "coverage": _frame_fingerprint(coverage) if coverage is not None else None,
        },
        "feature_rows": len(features),
        "feature_columns": list(features.columns),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = target.with_suffix(".manifest.json")
    descriptor, parquet_temp = tempfile.mkstemp(
        prefix=f".{target.stem}-", suffix=".parquet", dir=target.parent
    )
    os.close(descriptor)
    manifest_temp: str | None = None
    try:
        features.to_parquet(parquet_temp, engine="pyarrow", index=False)
        manifest["parquet_sha256"] = _file_sha256(parquet_temp)
        descriptor, manifest_temp = tempfile.mkstemp(
            prefix=f".{target.stem}-", suffix=".json", dir=target.parent
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(parquet_temp, target)
        os.replace(manifest_temp, manifest_path)
    finally:
        for temporary in (parquet_temp, manifest_temp):
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
    return features
