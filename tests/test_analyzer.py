"""Offline tests for the public FinBERT inference contract."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from news_sentiment import SentimentAnalyzer
from news_sentiment.results import PREDICTION_COLUMNS


class FakePipeline:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            config=SimpleNamespace(id2label={0: "Negative", 1: "Neutral", 2: "Positive"})
        )
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return [
            [
                {"label": "Positive", "score": 0.7},
                {"label": "Negative", "score": 0.1},
                {"label": "Neutral", "score": 0.2},
            ]
            for _ in texts
        ]


@pytest.fixture
def analyzer(monkeypatch):
    classifier = FakePipeline()
    created = []

    def make_pipeline(**kwargs):
        created.append(kwargs)
        return classifier

    class FakeBertConfig:
        @staticmethod
        def from_pretrained(model_name):
            assert model_name == "yiyanghkust/finbert-tone"
            return "legacy-bert-config"

    class FakeBertTokenizer:
        @staticmethod
        def from_pretrained(model_name):
            assert model_name == "yiyanghkust/finbert-tone"
            return "legacy-bert-tokenizer"

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            pipeline=make_pipeline,
            BertConfig=FakeBertConfig,
            BertTokenizer=FakeBertTokenizer,
        ),
    )
    instance = SentimentAnalyzer(device="cpu", batch_size=2)
    assert len(created) == 1
    assert created[0]["model"] == "yiyanghkust/finbert-tone"
    assert created[0]["config"] == "legacy-bert-config"
    assert created[0]["tokenizer"] == "legacy-bert-tokenizer"
    return instance, classifier, created


def test_predict_preserves_order_and_all_probabilities(analyzer):
    instance, classifier, created = analyzer
    texts = (text for text in ["first", "second"])
    result = instance.predict(texts, batch_size=1)
    assert list(result.columns) == list(PREDICTION_COLUMNS)
    assert result["text"].tolist() == ["first", "second"]
    assert result["label"].tolist() == ["positive", "positive"]
    assert result["sentiment_score"].tolist() == pytest.approx([0.6, 0.6])
    assert classifier.calls[0][1] == {
        "top_k": None,
        "truncation": True,
        "max_length": 512,
        "batch_size": 1,
    }
    instance.predict(["third"])
    assert len(created) == 1


def test_predict_one_and_label_id_mapping(analyzer):
    instance, classifier, _ = analyzer
    prediction = instance.predict_one("A company raised guidance")
    assert prediction.label == "positive"
    assert prediction.confidence == pytest.approx(0.7)
    assert classifier.calls[-1][1]["batch_size"] == 1
    assert instance._normalize_label("LABEL_0") == "negative"
    assert instance._normalize_label("label_2") == "positive"


@pytest.mark.parametrize("texts, error", [
    ("bare string", TypeError),
    ([], ValueError),
    (["valid", "  "], ValueError),
    (["valid", 3], TypeError),
])
def test_invalid_texts(analyzer, texts, error):
    with pytest.raises(error):
        analyzer[0].predict(texts)


@pytest.mark.parametrize("kwargs, error", [
    ({"model_name": "  "}, ValueError),
    ({"batch_size": 0}, ValueError),
    ({"max_length": 0}, ValueError),
    ({"batch_size": True}, TypeError),
])
def test_invalid_configuration(monkeypatch, kwargs, error):
    monkeypatch.setattr(SentimentAnalyzer, "_load_classifier", lambda self: FakePipeline())
    with pytest.raises(error):
        SentimentAnalyzer(**kwargs)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_FINBERT_SMOKE") != "1", reason="opt-in model download")
def test_real_finbert_checkpoint():
    analyzer = SentimentAnalyzer(device="cpu")
    result = analyzer.predict(["The company raised its annual guidance."])
    assert list(result.columns) == list(PREDICTION_COLUMNS)
    assert result.loc[0, ["p_negative", "p_neutral", "p_positive"]].sum() == pytest.approx(1.0)
