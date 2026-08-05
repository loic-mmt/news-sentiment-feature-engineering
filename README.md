# News Sentiment

Mini-librairie Python pour analyser le sentiment de textes financiers anglais avec
FinBERT. Elle retourne labels, probabilités, confiance, score continu, statistiques,
métriques d'évaluation et graphiques Matplotlib.

Modèle par défaut : [`yiyanghkust/finbert-tone`](https://huggingface.co/yiyanghkust/finbert-tone).

## Installation

Python 3.11 ou supérieur :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Alternative avec fichier de dépendances :

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

Le premier `SentimentAnalyzer()` télécharge le checkpoint Hugging Face. Les lancements
suivants utilisent le cache local. Le CPU est choisi par défaut sans GPU CUDA disponible.

## Utilisation rapide

```python
from news_sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()
prediction = analyzer.predict_one("The company raised its annual guidance.")

print(prediction.label)
print(prediction.confidence)
print(prediction.probabilities)
print(prediction.score)
```

`score` vaut `P(positive) - P(negative)` et reste compris entre -1 et 1.

## Analyse d'un lot

```python
texts = [
    "The company raised its annual guidance.",
    "Revenue remained unchanged from last year.",
    "The group reported a sharp decline in quarterly earnings.",
]

predictions = analyzer.predict(texts, batch_size=16)
print(predictions)
print(analyzer.summarize(predictions, confidence_threshold=0.60))
```

Colonnes produites :

| Colonne | Contenu |
|---|---|
| `text` | Texte original |
| `label` | `negative`, `neutral` ou `positive` |
| `confidence` | Plus grande probabilité |
| `p_negative` | Probabilité négative |
| `p_neutral` | Probabilité neutre |
| `p_positive` | Probabilité positive |
| `sentiment_score` | `p_positive - p_negative` |

## Évaluation

Des labels réels sont requis :

```python
y_true = ["positive", "neutral", "negative"]
report = analyzer.evaluate(predictions, y_true)

print(report.accuracy)
print(report.macro_f1)
print(report.per_class)
print(report.confusion_matrix)
```

## Graphiques

Les fonctions retournent un `matplotlib.axes.Axes`. Elles n'affichent et ne sauvegardent
rien automatiquement.

```python
import matplotlib.pyplot as plt
import pandas as pd

analyzer.plot_labels(predictions, proportions=True)
analyzer.plot_score_distribution(predictions)
analyzer.plot_confusion_matrix(report, normalize=True)

predictions["published_at"] = pd.to_datetime(
    ["2026-08-01", "2026-08-02", "2026-08-03"]
)
analyzer.plot_timeline(predictions, date_col="published_at", freq="D")

plt.show()
```

## Device et configuration

```python
cpu_analyzer = SentimentAnalyzer(device="cpu")
gpu_analyzer = SentimentAnalyzer(device="cuda:0", batch_size=64, max_length=512)
```

Valeurs acceptées pour `device` : `None`, `-1`, index GPU entier, `"cpu"`,
`"cuda"` ou `"cuda:<index>"`.

## Données

Aucune donnée nécessaire pour faire une prédiction. Pour évaluation annotée ou série
temporelle, voir [DATA.md](DATA.md). Les enseignements tirés des expériences du cours,
les critères de fine-tuning et les pièges d'évaluation sont conservés dans
[MODEL_NOTES.md](MODEL_NOTES.md).

## Limites du MVP

- Textes financiers anglais.
- Aucun entraînement ou fine-tuning.
- Aucun serveur HTTP ou stockage.
- Probabilités non calibrées sur les données propres à l'utilisateur.
- Les résultats dépendent du domaine et du checkpoint choisi.
