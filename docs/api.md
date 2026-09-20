# API de la librairie

Tous les symboles ci-dessous s'importent depuis `news_sentiment`. Les timestamps
fournis par l'appelant doivent avoir un fuseau explicite ; les résultats sont
normalisés en UTC.

## Collecte et stockage

```python
from news_sentiment import fetch_rss, fetch_alpha_vantage, save_news
```

- `fetch_rss(feed_urls, *, ticker_aliases=None, fetched_at=None,
  timeout_seconds=10)` lit des flux RSS/Atom configurés par l'appelant. Les alias
  explicites sont cherchés dans titre et résumé. Un article multi-ticker produit
  une ligne par ticker ; un article sans correspondance conserve un ticker manquant.
- `fetch_alpha_vantage(*, api_key=None, tickers=None, topics=None,
  time_from=None, time_to=None, sort="LATEST", limit=50, fetched_at=None,
  timeout_seconds=10, quota_path=None, hourly_limit=25, daily_limit=25)` lit une
  page `NEWS_SENTIMENT`. `api_key` est sinon pris dans
  `ALPHAVANTAGE_API_KEY`. Une liste de tickers est un filtre **ET** côté API, et
  chaque appel tenté est compté avant l'envoi. Les plafonds sont locaux et
  configurables ; `None` les désactive individuellement.
- `save_news(news, path)` fusionne les nouvelles lignes dans un fichier Parquet,
  déduplique par `(news_id, ticker)` et préserve le premier `available_at`. Il
  retourne le DataFrame cumulé. Une seule écriture simultanée par fichier est
  supposée ; coordonner plusieurs producteurs en amont si nécessaire.

Schéma normalisé commun : `news_id`, `published_at`, `available_at`, `source`,
`url`, `ticker`, `title`, `text`. `published_at` vient de la source ;
`available_at` est la première observation **par ce pipeline**, jamais une
reconstruction à partir de la publication. Les colonnes `av_*` d'Alpha Vantage
sont des métadonnées distinctes des scores FinBERT.

Le paramètre `fetched_at` permet de fournir une heure d'observation connue
(notamment en test). En production, ne l'antidatez pas : cela introduirait des
news dans des décisions antérieures à leur première collecte réelle.

## Prédiction et évaluation

`SentimentAnalyzer(model_name=DEFAULT_MODEL, *, device=None, batch_size=32,
max_length=512)` charge le checkpoint à la première prédiction. `predict_one(text)`
retourne un objet `Prediction` avec `label`, `probabilities`, `confidence` et
`score`. `predict(texts, *, batch_size=None)` retourne un DataFrame dans l'ordre
d'entrée, avec `text`, `label`, `confidence`, `p_negative`, `p_neutral`,
`p_positive` et `sentiment_score = p_positive - p_negative`.

`summarize(predictions, *, confidence_threshold=0.60)` retourne des statistiques
descriptives. `evaluate(predictions, y_true)` exige des labels de référence et
retourne `EvaluationReport` (`accuracy`, `macro_f1`, métriques par classe et
matrice de confusion). Les méthodes équivalentes existent sur l'analyseur.
`plot_labels`, `plot_score_distribution`, `plot_timeline` et
`plot_confusion_matrix` retournent un `matplotlib.axes.Axes` sans appeler
`show()` ni sauvegarder de fichier.

## Sentiment et features

```python
from news_sentiment import (
    SentimentAnalyzer, attach_sentiment, build_sentiment_features,
)

analyzer = SentimentAnalyzer()
scored = attach_sentiment(news, analyzer)
features = build_sentiment_features(scored, decision_points)
```

`attach_sentiment(news, analyzer, *, text_col="text")` appelle une fois
`analyzer.predict()` pour le lot et conserve colonnes, index et ordre des news.
Il ajoute `label`, `confidence`, `p_negative`, `p_neutral`, `p_positive` et
`sentiment_score = p_positive - p_negative`.

`decision_points` doit contenir des paires uniques `(ticker, decision_at)` avec
fuseau. `build_sentiment_features(scored_news, decision_points, *,
lookback="24h", short_lookback="6h", half_life="6h")` produit exactement une
ligne par paire, dans le même ordre. Pour chaque décision à `t`, seules les news
du ticker avec `available_at ∈ (t - lookback, t]` contribuent aux agrégats :

| Feature | Sens |
|---|---|
| `news_count` | Nombre de news uniques dans la fenêtre |
| `sentiment_mean`, `sentiment_std` | Moyenne et écart-type population du score |
| `p_positive_mean`, `p_neutral_mean`, `p_negative_mean` | Probabilités moyennes |
| `positive_share`, `negative_share` | Parts de labels |
| `confidence_mean` | Confiance moyenne |
| `sentiment_ewm` | Score pondéré par récence et confiance |
| `sentiment_momentum` | Moyenne courte moins moyenne complète |
| `hours_since_last_news` | Âge de la dernière news déjà connue, même hors fenêtre |

Sans news dans la fenêtre, `news_count=0` et les mesures non définies sont
manquantes. La librairie ne choisit ni calendrier de marché ni heure de décision.

## Quota Alpha Vantage

```python
import os
from news_sentiment import AlphaVantageQuota

status = AlphaVantageQuota(os.environ["ALPHAVANTAGE_API_KEY"]).status()
print(status.hour_used, status.hour_remaining)
print(status.day_used, status.day_remaining)
```

La CLI équivalente est `news-sentiment alpha-vantage quota`. Par défaut, le
compteur SQLite est dans
`~/.local/share/news_sentiment/alpha_vantage_quota.sqlite3` ; le chemin se
change avec `NEWS_SENTIMENT_QUOTA_DB`, `quota_path=` ou `--quota-db`. Seule une
empreinte de la clé est stockée. Les fenêtres sont glissantes sur 1h et 24h ;
les appels faits hors de cette librairie ou depuis un autre compteur ne sont pas
visibles. Une tentative échouée reste comptée par prudence.

Pour un parcours complet, voir le [README](../README.md).
