# API de la librairie

Tous les symboles ci-dessous s'importent depuis `news_sentiment`. Les timestamps
fournis par l'appelant doivent avoir un fuseau explicite ; les résultats sont
normalisés en UTC.

## Collecte et stockage

```python
from news_sentiment import (
    fetch_rss, fetch_alpha_vantage, import_historical_news,
    save_news, save_coverage, validate_coverage,
)
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
  déduplique par `(news_id, ticker)` et préserve la première ligne déjà stockée,
  y compris son `available_at` si une relance prétend une date plus ancienne. Il
  retourne le DataFrame cumulé. Une seule écriture simultanée par fichier est
  supposée ; coordonner plusieurs producteurs en amont si nécessaire.

Schéma normalisé commun : `news_id`, `published_at`, `available_at`, `source`,
`url`, `ticker`, `title`, `text`. `published_at` vient de la source ;
`available_at` est l'heure à laquelle l'article était disponible, jamais une
reconstruction à partir de la publication. Les connecteurs du pipeline y
mettent leur heure de première observation, avec
`availability_kind="pipeline_observed"` et une `availability_reference` au
connecteur. Les colonnes `av_*` d'Alpha Vantage sont des métadonnées distinctes
des scores FinBERT.

Le paramètre `fetched_at` permet de fournir une heure d'observation connue
(notamment en test). En production, ne l'antidatez pas : cela introduirait des
news dans des décisions antérieures à leur première collecte réelle.

### Historique vérifiable

`import_historical_news(articles)` exige les huit colonnes normalisées plus
`availability_kind` (`provider_first_seen` ou `archive_first_seen`) et
`availability_reference` (référence non vide au champ fournisseur ou snapshot
d'archive). `news_id` et `ticker` doivent identifier des paires uniques,
`source` ne peut être vide, `published_at` et `available_at` doivent avoir un
fuseau, et `available_at` ne peut précéder `published_at`. La fonction retourne
un DataFrame validé à passer à `save_news()` puis `attach_sentiment()`.

La librairie valide l'existence et la cohérence de la **déclaration** de
provenance ; elle ne peut pas vérifier seule une preuve externe. Une collecte
faite aujourd'hui reste `pipeline_observed`, même si l'article a été publié
il y a des années. Un import historique vérifié doit avoir son propre fichier
Parquet : `save_news()` ne réécrit jamais silencieusement une paire déjà stockée.

### Journal de couverture séparé

`validate_coverage(journal)` contrôle les colonnes `source`, `ticker` (manquant
pour tous les tickers), `start_at`, `end_at`, `recorded_at`, `status` et
`evidence_reference`. Les intervalles sont `[start_at, end_at)` en UTC ; les
statuts enregistrables sont `covered` et `incomplete`. `recorded_at` est l'heure
à laquelle l'assertion de couverture était disponible. `save_coverage(journal,
path)` la déduplique dans un Parquet distinct des news et retourne le journal
cumulé.

`source` est un identifiant stable de flux ou fournisseur choisi par l'appelant,
pas forcément le champ `source` des articles, qui peut désigner leur éditeur.
Ni le succès d'une collecte RSS, ni une page Alpha Vantage ne prouvent à eux
seuls la couverture complète d'une fenêtre historique.

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
lookback="24h", short_lookback="6h", half_life="6h",
include_at_cutoff=True, coverage=None, required_sources=None)` produit exactement une
ligne par paire, dans le même ordre. Pour chaque décision à `t`, seules les news
du ticker avec `available_at ∈ (t - lookback, t]` contribuent aux agrégats. Mettre
`include_at_cutoff=False` pour imposer `available_at < t` :

| Feature | Sens |
|---|---|
| `news_count` | Nombre de news uniques dans la fenêtre |
| `coverage_status` | `covered`, `incomplete` ou `unknown` |
| `last_contributing_available_at` | Dernière disponibilité ayant contribué aux agrégats, ou `NaT` |
| `sentiment_mean`, `sentiment_std` | Moyenne et écart-type population du score |
| `p_positive_mean`, `p_neutral_mean`, `p_negative_mean` | Probabilités moyennes |
| `positive_share`, `negative_share` | Parts de labels |
| `confidence_mean` | Confiance moyenne |
| `sentiment_ewm` | Score pondéré par récence et confiance |
| `sentiment_momentum` | Moyenne courte moins moyenne complète |
| `hours_since_last_news` | Âge de la dernière news déjà connue, même hors fenêtre |

Sans news dans la fenêtre, `news_count=0` et les mesures non définies sont
manquantes. `hours_since_last_news` peut rester défini si une news plus ancienne
était connue ; ce n'est pas `last_contributing_available_at`. Sans journal,
`coverage_status="unknown"`. Avec journal, `required_sources` doit énumérer les
sources attendues. Une assertion enregistrée après la décision est ignorée ;
`covered` exige que toutes les sources couvrent toute la fenêtre, `incomplete`
signale une lacune explicite, et les preuves absentes/partielles donnent
`unknown`. Pour chaque portion d'intervalle, la dernière assertion connue
prévaut : une correction ultérieure peut rétablir `covered` pour les décisions
futures, sans modifier les décisions antérieures. La librairie ne choisit ni
calendrier de marché ni heure de décision.

### Export auditable

`export_sentiment_features(scored_news, decision_points, path, *, checkpoint,
input_identifiers=None, coverage=None, required_sources=None, lookback="24h",
short_lookback="6h", half_life="6h", include_at_cutoff=True)` calcule les mêmes
features puis écrit `path` en Parquet et un sidecar `.manifest.json`. Le
manifeste contient version du schéma, fenêtres et borne, checkpoint, sources
requises, identifiants fournis par l'appelant, nombre de lignes et SHA-256 des
trois entrées, puis SHA-256 du Parquet. Une discordance de checksum révèle une
paire Parquet/manifeste incohérente. L'export exige des news normalisées avec
`availability_kind` et `availability_reference` valides ; contrairement au calcul
en mémoire, il refuse un DataFrame sans provenance. Ne placer aucun secret dans
`input_identifiers`. Pour une reproduction exacte, employer un checkpoint épinglé
à une révision immuable et conserver les entrées identifiées.

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
