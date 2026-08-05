"""FinBERT model loading and sentiment inference."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd
import numpy as np
from matplotlib.axes import Axes
import torch

from .results import (
    PREDICTION_COLUMNS,
    SENTIMENT_LABELS,
    EvaluationReport,
    Prediction,
    SentimentLabel,
)
from .stats import SentimentSummary

DEFAULT_MODEL = "yiyanghkust/finbert-tone"

_DIRECT_LABELS: dict[str, SentimentLabel] = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}

# Ordre conseillé pour implémenter ce module :
# 1. _materialize_texts
# 2. _normalize_label
# 3. _prediction_from_scores
# 4. _load_classifier
# 5. predict_one puis predict
# Référence principale : course/03_sentiment_evolution.py::get_finbert_predictions
# Exemple d'inférence en lot : course/07_news_return_signals.py::score_sentiment_finbert


class SentimentAnalyzer:
    """Analyze financial sentiment with a Hugging Face classification model.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier or local model directory.
    device:
        ``None`` for automatic selection, ``"cpu"``, ``"cuda"`` or a device
        index accepted by Transformers.
    batch_size:
        Default number of texts processed in one inference batch.
    max_length:
        Maximum number of tokens kept for each input.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | int | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        if not isinstance(model_name, str):
            raise TypeError(
                f"model_name must be a string, got {type(model_name).__name__}"
            )
        normalized_model_name = model_name.strip()
        if not normalized_model_name:
            raise ValueError("model_name must not be empty or blank")

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be greater than or equal to 1")

        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError("max_length must be an integer")
        if max_length < 1:
            raise ValueError("max_length must be greater than or equal to 1")

        self.model_name = normalized_model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        # Kept untyped because the concrete Transformers classes depend on the
        # selected checkpoint.
        self._classifier: Any | None = None
        self._classifier = self._load_classifier()

    def _load_classifier(self) -> Any:
        """Create and return the configured Transformers pipeline."""
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required to load the sentiment classifier"
            ) from exc

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = self.model_name.strip()

        requested_device = self.device
        if requested_device is None:
            resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif isinstance(requested_device, bool):
            raise TypeError("device must not be a boolean")
        elif isinstance(requested_device, int):
            if requested_device < -1:
                raise ValueError("an integer device must be -1 or a non-negative index")
            resolved_device = (
                "cpu" if requested_device == -1 else f"cuda:{requested_device}"
            )
        elif isinstance(requested_device, str):
            normalized_device = requested_device.strip().lower()
            if normalized_device == "cpu":
                resolved_device = "cpu"
            elif normalized_device == "cuda":
                resolved_device = "cuda:0"
            elif normalized_device.startswith("cuda:"):
                index_text = normalized_device.removeprefix("cuda:")
                if not index_text.isdecimal():
                    raise ValueError(
                        "a CUDA device must use the format "
                        "'cuda:<non-negative index>'"
                    )
                resolved_device = f"cuda:{int(index_text)}"
            else:
                raise ValueError(
                    "device must be None, -1, a non-negative integer, 'cpu', "
                    "'cuda' or 'cuda:<index>'"
                )
        else:
            raise TypeError(
                "device must be None, an integer or a string, got "
                f"{type(requested_device).__name__}"
            )

        if resolved_device.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device {resolved_device!r} was requested but CUDA is unavailable"
                )
            device_index = int(resolved_device.removeprefix("cuda:"))
            device_count = torch.cuda.device_count()
            if device_index >= device_count:
                raise ValueError(
                    f"CUDA device index {device_index} is unavailable; "
                    f"found {device_count} CUDA device(s)"
                )

        classifier = pipeline(
            task="sentiment-analysis",
            model=self.model_name,
            tokenizer=self.model_name,
            device=resolved_device,
        )

        config = getattr(getattr(classifier, "model", None), "config", None)
        id2label = getattr(config, "id2label", None)
        if not isinstance(id2label, Mapping) or len(id2label) != len(SENTIMENT_LABELS):
            raise ValueError(
                f"model {self.model_name!r} must provide exactly three labels "
                "through config.id2label"
            )

        semantic_labels: set[SentimentLabel] = set()
        for label_id, raw_label in id2label.items():
            if not isinstance(raw_label, str):
                raise ValueError(
                    f"id2label[{label_id!r}] must be a string, "
                    f"got {type(raw_label).__name__}"
                )
            semantic_label = _DIRECT_LABELS.get(raw_label.strip().casefold())
            if semantic_label is None:
                raise ValueError(
                    f"id2label[{label_id!r}]={raw_label!r} does not identify a "
                    "negative, neutral or positive sentiment"
                )
            semantic_labels.add(semantic_label)

        if semantic_labels != set(SENTIMENT_LABELS):
            raise ValueError(
                f"model {self.model_name!r} must expose each sentiment exactly once: "
                f"{list(SENTIMENT_LABELS)}"
            )

        self.device = resolved_device
        return classifier


    def _normalize_label(self, raw_label: str) -> SentimentLabel:
        """Map a model-specific label to the public lowercase label set."""
        if not isinstance(raw_label, str):
            raise TypeError(
                f"raw_label must be a string, got {type(raw_label).__name__}"
            )

        normalized = raw_label.strip().casefold()
        if not normalized:
            raise ValueError("raw_label must not be empty or blank")

        direct_label = _DIRECT_LABELS.get(normalized)
        if direct_label is not None:
            return direct_label

        prefix = "label_"
        label_id_text = normalized.removeprefix(prefix)
        if not normalized.startswith(prefix) or not label_id_text.isdecimal():
            raise ValueError(
                f"unsupported sentiment label {raw_label!r} for model "
                f"{self.model_name!r}"
            )

        if self._classifier is None:
            raise RuntimeError(
                f"cannot resolve {raw_label!r}: the classifier is not loaded"
            )

        config = getattr(getattr(self._classifier, "model", None), "config", None)
        id2label = getattr(config, "id2label", None)
        if not isinstance(id2label, Mapping):
            raise ValueError(
                f"model {self.model_name!r} does not provide a valid id2label mapping"
            )

        label_id = int(label_id_text)
        mapped_label = id2label.get(label_id)
        if mapped_label is None:
            mapped_label = id2label.get(str(label_id))
        if mapped_label is None:
            raise ValueError(
                f"label id {label_id} is missing from the id2label mapping of "
                f"model {self.model_name!r}"
            )
        if not isinstance(mapped_label, str):
            raise ValueError(
                f"id2label[{label_id}] must be a string, "
                f"got {type(mapped_label).__name__}"
            )

        semantic_label = _DIRECT_LABELS.get(mapped_label.strip().casefold())
        if semantic_label is None:
            raise ValueError(
                f"id2label[{label_id}]={mapped_label!r} does not identify a "
                "negative, neutral or positive sentiment"
            )

        return semantic_label

    @staticmethod
    def _materialize_texts(texts: Iterable[str]) -> list[str]:
        """Materialize and validate an iterable of non-empty texts."""
        if isinstance(texts, str):
            raise TypeError(
                "texts must be an iterable of strings; use predict_one(text) "
                "for a single string"
            )

        try:
            materialized = list(texts)
        except TypeError as exc:
            raise TypeError("texts must be an iterable of strings") from exc

        if not materialized:
            raise ValueError("texts must contain at least one text")

        for index, text in enumerate(materialized):
            if not isinstance(text, str):
                raise TypeError(
                    f"texts[{index}] must be a string, got {type(text).__name__}"
                )
            if not text.strip():
                raise ValueError(f"texts[{index}] must not be empty or blank")

        return materialized

    def _prediction_from_scores(
        self,
        text: str,
        raw_scores: Sequence[dict[str, Any]],
    ) -> Prediction:
        """Build a normalized ``Prediction`` from one model response."""
        # Format attendu de raw_scores avec top_k=None :
        # [{"label": "Neutral", "score": 0.80}, ... trois éléments au total].
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("text must not be empty or blank")

        if isinstance(raw_scores, (str, bytes)) or not isinstance(raw_scores, Sequence):
            raise TypeError("raw_scores must be a sequence of score dictionaries")
        if len(raw_scores) == 0:
            raise ValueError("raw_scores must not be empty")

        probabilities: dict[SentimentLabel, float] = {}

        for index, raw_result in enumerate(raw_scores):
            if not isinstance(raw_result, dict):
                raise TypeError(
                    f"raw_scores[{index}] must be a dictionary, "
                    f"got {type(raw_result).__name__}"
                )

            missing_keys = {"label", "score"} - raw_result.keys()
            if missing_keys:
                raise ValueError(
                    f"raw_scores[{index}] is missing keys: {sorted(missing_keys)}"
                )

            normalized_label = self._normalize_label(raw_result["label"])
            if normalized_label not in SENTIMENT_LABELS:
                raise ValueError(
                    f"raw_scores[{index}] contains an unsupported label: "
                    f"{normalized_label!r}"
                )

            try:
                probability = float(raw_result["score"])
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"raw_scores[{index}]['score'] must be a real number"
                ) from exc

            if not np.isfinite(probability):
                raise ValueError(f"raw_scores[{index}]['score'] must be finite")
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"raw_scores[{index}]['score'] must be between 0 and 1"
                )
            if normalized_label in probabilities:
                raise ValueError(f"duplicate sentiment label: {normalized_label}")

            probabilities[normalized_label] = probability

        missing_labels = [
            label for label in SENTIMENT_LABELS if label not in probabilities
        ]
        if missing_labels:
            raise ValueError(f"raw_scores is missing labels: {missing_labels}")

        probability_sum = sum(probabilities.values())
        if not np.isclose(probability_sum, 1.0, rtol=0.0, atol=1e-5):
            raise ValueError(
                f"probabilities must sum to 1, got {probability_sum:.8f}"
            )

        # max() keeps the first item on ties, so SENTIMENT_LABELS defines the
        # deterministic tie-breaking order.
        winning_label = max(SENTIMENT_LABELS, key=probabilities.__getitem__)
        confidence = probabilities[winning_label]
        score = probabilities["positive"] - probabilities["negative"]

        return Prediction(
            text=text,
            label=winning_label,
            confidence=confidence,
            probabilities=probabilities,
            score=score,
        )

    def _predict_many(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> list[Prediction]:
        """Run one validated batch inference and preserve input order."""
        if self._classifier is None:
            self._classifier = self._load_classifier()

        raw_results = self._classifier(
            list(texts),
            top_k=None,
            truncation=True,
            max_length=self.max_length,
            batch_size=batch_size,
        )

        if isinstance(raw_results, (str, bytes)) or not isinstance(
            raw_results, Sequence
        ):
            raise TypeError(
                "the classifier must return one sequence of scores per input text"
            )
        if len(raw_results) != len(texts):
            raise ValueError(
                "classifier output length mismatch: "
                f"expected {len(texts)}, got {len(raw_results)}"
            )

        return [
            self._prediction_from_scores(text, raw_scores)
            for text, raw_scores in zip(texts, raw_results, strict=True)
        ]

    def predict_one(self, text: str) -> Prediction:
        """Predict the sentiment of one text."""
        materialized_text = self._materialize_texts([text])[0]
        return self._predict_many([materialized_text], batch_size=1)[0]

    def predict(
        self,
        texts: Iterable[str],
        *,
        batch_size: int | None = None,
    ) -> pd.DataFrame:
        """Predict a batch and return one flat DataFrame row per input text.

        The output columns are ``text``, ``label``, ``confidence``,
        ``p_negative``, ``p_neutral``, ``p_positive`` and ``sentiment_score``.
        Input order must be preserved.
        """
        materialized_texts = self._materialize_texts(texts)
        effective_batch_size = self.batch_size if batch_size is None else batch_size

        if isinstance(effective_batch_size, bool) or not isinstance(
            effective_batch_size, int
        ):
            raise TypeError("batch_size must be an integer")
        if effective_batch_size < 1:
            raise ValueError("batch_size must be greater than or equal to 1")

        predictions = self._predict_many(
            materialized_texts,
            batch_size=effective_batch_size,
        )
        records = [prediction.to_record() for prediction in predictions]

        return pd.DataFrame.from_records(records, columns=PREDICTION_COLUMNS)

    def summarize(
        self,
        predictions: pd.DataFrame,
        *,
        confidence_threshold: float = 0.60,
    ) -> SentimentSummary:
        """Return descriptive statistics for a prediction DataFrame."""
        from .stats import summarize

        return summarize(predictions, confidence_threshold=confidence_threshold)

    def evaluate(
        self,
        predictions: pd.DataFrame,
        y_true: Iterable[str],
    ) -> EvaluationReport:
        """Evaluate a prediction DataFrame against known labels."""
        from .stats import evaluate

        return evaluate(predictions, y_true)

    def plot_labels(
        self,
        predictions: pd.DataFrame,
        *,
        proportions: bool = False,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot prediction counts or proportions by label."""
        from .plots import plot_labels

        return plot_labels(predictions, proportions=proportions, ax=ax)

    def plot_score_distribution(
        self,
        predictions: pd.DataFrame,
        *,
        bins: int = 30,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot the distribution of the continuous sentiment score."""
        from .plots import plot_score_distribution

        return plot_score_distribution(predictions, bins=bins, ax=ax)

    def plot_timeline(
        self,
        predictions: pd.DataFrame,
        *,
        date_col: str,
        freq: str = "D",
        ax: Axes | None = None,
    ) -> Axes:
        """Plot average sentiment over time."""
        from .plots import plot_timeline

        return plot_timeline(predictions, date_col=date_col, freq=freq, ax=ax)

    def plot_confusion_matrix(
        self,
        report: EvaluationReport,
        *,
        normalize: bool = False,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot an evaluation report's confusion matrix."""
        from .plots import plot_confusion_matrix

        return plot_confusion_matrix(report, normalize=normalize, ax=ax)
