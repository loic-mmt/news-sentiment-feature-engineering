# News Sentiment

Bibliothèque Python pour transformer des news financières en features de sentiment
par ticker et instant de décision. Elle collecte des flux RSS/Atom configurés par
l'utilisateur, interroge éventuellement Alpha Vantage, ou accepte un `DataFrame`
existant. Le texte est scoré avec le checkpoint FinBERT pré-entraîné
[`yiyanghkust/finbert-tone`](https://huggingface.co/yiyanghkust/finbert-tone).

La librairie ne fournit ni flux prédéfini, ni scraping de pages, ni modèle de
rendements ou de trading. Python 3.10 ou supérieur.

## Installation

Depuis GitHub :

```bash
python -m pip install "news-sentiment-feature-engineering[news] @ git+https://github.com/loic-mmt/news-sentiment-feature-engineering.git"
```

Pour contribuer depuis un clone :

```bash
git clone https://github.com/loic-mmt/news-sentiment-feature-engineering.git
cd news-sentiment-feature-engineering
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[news,dev]"
```

L'extra `[news]` installe `feedparser` pour RSS/Atom et `pyarrow` pour Parquet.
La prédiction seule fonctionne sans cet extra. Le premier appel à
`SentimentAnalyzer()` télécharge le checkpoint Hugging Face ; ses poids ne sont
pas inclus dans le dépôt.

Ce projet n'est **pas publié sur PyPI** : utilisez l'URL GitHub ci-dessus, pas
`pip install news-sentiment-feature-engineering` seul. Le nom d'import reste
`news_sentiment` et la commande CLI reste `news-sentiment`.

## RSS → FinBERT → features

```python
import pandas as pd
from news_sentiment import (
    SentimentAnalyzer,
    attach_sentiment,
    build_sentiment_features,
    export_sentiment_features,
    fetch_rss,
    save_news,
)

news = fetch_rss(
    ["https://votre-source.example/finance.xml"],
    ticker_aliases={"AAPL": ["Apple"], "MSFT": ["Microsoft"]},
)
stored = save_news(news, "data/news.parquet")
analyzer = SentimentAnalyzer()
scored = attach_sentiment(stored, analyzer)
decision_points = pd.DataFrame({
    "ticker": ["AAPL", "MSFT"],
    "decision_at": [pd.Timestamp.now(tz="UTC")] * 2,
})
features = export_sentiment_features(
    scored, decision_points, "data/features.parquet",
    checkpoint=analyzer.model_name,
    input_identifiers={"news_store": "data/news.parquet"},
    include_at_cutoff=False,
)
print(features)
```

`fetch_rss()` utilise titres et résumés des flux, sans télécharger les articles.
Les tickers sont associés uniquement par les alias fournis ; un article peut
produire plusieurs lignes ou rester sans ticker. `save_news()` fusionne les
collectes par `(news_id, ticker)` et conserve le premier `available_at` stocké,
même si une relance prétend avoir observé l'article plus tôt. Les lignes collectées
portent `availability_kind="pipeline_observed"` et la référence du connecteur.

`decision_points` définit exactement les paires `(ticker, decision_at)` en sortie.
Pour chaque paire, seules les news observées dans
`(decision_at - lookback, decision_at]` contribuent aux agrégats par défaut.
`include_at_cutoff=False` impose la borne stricte
`(decision_at - lookback, decision_at)`, utile pour une décision à minuit UTC. Par défaut,
`lookback="24h"`, `short_lookback="6h"` et `half_life="6h"`. Une paire sans
news dans sa fenêtre conserve `news_count=0` ; les mesures non définies sont
manquantes. `hours_since_last_news` peut rester défini grâce à une news plus
ancienne. `last_contributing_available_at` identifie la dernière news dans la
fenêtre. Sans journal de couverture, `coverage_status="unknown"` : zéro news ne
prouve pas une collecte complète. Toutes les dates fournies par l'appelant
doivent avoir un fuseau ; les résultats sont normalisés en UTC.

`export_sentiment_features()` écrit le Parquet et un fichier
`features.manifest.json` contenant version de schéma, fenêtres, borne, checkpoint,
identifiants et empreintes des entrées, puis empreinte du Parquet. Pour un audit
reproductible, fournir un identifiant de checkpoint immuable (révision ou digest)
et conserver les entrées correspondantes. L'export refuse des news sans schéma
normalisé ni provenance explicite de `available_at` ; le calcul en mémoire reste
possible avec un DataFrame minimal.

Un [exemple exécutable](examples/realtime_features.py) lit les URL RSS, un JSON
d'alias et un CSV de décisions, puis écrit news et features en Parquet. Les
colonnes et contrats complets figurent dans la [documentation API](docs/api.md).

## Historique et couverture des sources

Pour importer un historique, utiliser `import_historical_news()` avant
`save_news()`. En plus du schéma normalisé, chaque article exige
`availability_kind` (`provider_first_seen` ou `archive_first_seen`) et une
`availability_reference` non vide pointant vers la preuve externe (champ du
fournisseur, identifiant de snapshot, etc.). `published_at` et `available_at`
doivent être horodatés avec fuseau ; `available_at` ne peut pas être déduit de
`published_at`. La validation contrôle le contrat, **pas la véracité de la
source**. Ne pas antidater `fetched_at` pour simuler un historique.

Le journal `save_coverage()` est un Parquet distinct des articles. Ses lignes
indiquent `source`, `ticker` facultatif, intervalle `[start_at, end_at)`,
`recorded_at`, `status` (`covered` ou `incomplete`) et `evidence_reference`.
Passer ce journal à `build_sentiment_features(..., coverage=journal,
required_sources=[...])` ou à l'export. `covered` n'est produit que si **chaque**
source requise couvre toute la fenêtre, avec une preuve connue avant la décision ;
une lacune donne `unknown`, une collecte explicitement incomplète donne
`incomplete`. La réussite d'un simple appel RSS ou d'une page Alpha Vantage
ne suffit pas à déclarer une période `covered`.

## Alpha Vantage et quota local

Le connecteur `fetch_alpha_vantage()` utilise l'endpoint
[`NEWS_SENTIMENT`](https://www.alphavantage.co/documentation/). Ses colonnes
`av_*` sont des métadonnées du fournisseur, distinctes des scores FinBERT.
Un filtre `tickers="AAPL,MSFT"` demande des articles mentionnant **les deux**
tickers ; pour deux recherches indépendantes, faire deux appels.

La clé doit être passée via `api_key=` ou la variable d'environnement
`ALPHAVANTAGE_API_KEY` (majuscules). La librairie ne charge pas `.env` ou
`secrets.env` automatiquement. Exemple dans un projet consommateur :

```bash
python -m pip install python-dotenv
```

```python
from dotenv import load_dotenv
from news_sentiment import fetch_alpha_vantage, save_news

load_dotenv(".env")
news = fetch_alpha_vantage(tickers="AAPL", limit=100)
stored = save_news(news, "data/news.parquet")
```

Utiliser [.env.example](.env.example) comme modèle et exclure le fichier de clé
du dépôt consommateur. Pour la CLI, la variable doit être disponible dans **son**
processus : un `load_dotenv()` lancé dans un autre processus ne suffit pas.

```bash
news-sentiment alpha-vantage quota
```

Cette commande lit le compteur SQLite local, sans appel réseau. Chaque tentative
d'appel est comptée avant l'envoi, même en cas d'échec. Le plafond quotidien par
défaut est de 25 appels sur 24 h glissantes ; celui de 25 appels sur 1 h est une
**politique locale**, pas une limite horaire publiée par Alpha Vantage. Les
plafonds sont configurables. Le compteur ne connaît pas les appels faits ailleurs
et n'est donc pas un solde fournisseur garanti. Voir la [documentation API](docs/api.md).

## Autres fonctions

`SentimentAnalyzer.predict_one()` retourne label, probabilités, confiance et
score continu `P(positive) - P(negative)`. `predict()` traite un lot ;
`summarize()`, `evaluate()` et les fonctions `plot_*` couvrent description,
évaluation sur labels réels et visualisation. Elles n'entraînent pas FinBERT.

```python
from news_sentiment import SentimentAnalyzer

prediction = SentimentAnalyzer().predict_one(
    "The company raised its annual guidance."
)
print(prediction.label, prediction.score)
```

## Vérification

Après installation avec `.[news,dev]`, `python -m pytest -q` lance les tests
hors ligne. La CI les exécute sous Python 3.10 à 3.13 et vérifie le paquet
construit. Deux tests d'intégration sont opt-in :

```bash
RUN_FINBERT_SMOKE=1 python -m pytest -q -m slow
RUN_ALPHA_VANTAGE_SMOKE=1 python -m pytest -q --tb=line -m slow tests/test_alpha_vantage.py
```

Le second exige `ALPHAVANTAGE_API_KEY` dans l'environnement et consomme une
tentative dans le quota local. Sur une installation Python macOS sans certificats
CA configurés, `CERTIFICATE_VERIFY_FAILED` nécessite de corriger le magasin de
certificats ou de définir `SSL_CERT_FILE` avant les appels HTTPS.

## Limites et licence

- FinBERT fourni ici cible des textes financiers anglais ; ses probabilités ne
  sont pas calibrées sur vos sources.
- Pas de calendrier de marché, d'heure de décision imposée ni de backtest.
- Le fichier Parquet suppose une seule écriture simultanée ; coordonner plusieurs
  producteurs en amont.
- `attach_sentiment()` re-score tout le lot fourni ; conserver les scores déjà
  calculés si le volume ou le coût d'inférence le justifie.

Code sous [licence MIT](LICENSE). Les poids du modèle et les données des
fournisseurs ne sont pas redistribués ; vérifier leurs propres conditions
d'utilisation.
