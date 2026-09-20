"""Offline CLI quota reporting tests."""

from __future__ import annotations

from news_sentiment.cli import main
from news_sentiment.quota import AlphaVantageQuota


def test_quota_cli_reads_same_ledger_without_network(monkeypatch, tmp_path, capsys):
    path = tmp_path / "usage.sqlite3"
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-secret")
    AlphaVantageQuota("test-secret", path=path).reserve()
    assert main(["alpha-vantage", "quota", "--quota-db", str(path)]) == 0
    output = capsys.readouterr().out
    assert "1h: 1/25 appels ; reste 24" in output
    assert "24h: 1/25 appels ; reste 24" in output
    assert "test-secret" not in output
    assert main(["alpha-vantage", "quota", "--quota-db", str(path), "--daily-limit", "unlimited"]) == 0
    assert "plafond local désactivé" in capsys.readouterr().out
    assert AlphaVantageQuota("test-secret", path=path).status().day_used == 1


def test_quota_cli_requires_environment_key(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    assert main(["alpha-vantage", "quota", "--quota-db", str(tmp_path / "unused.sqlite3")]) == 2
    assert "ALPHAVANTAGE_API_KEY" in capsys.readouterr().err
