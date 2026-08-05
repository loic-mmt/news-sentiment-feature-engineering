# Données disponibles

La librairie n'a besoin d'aucun dataset pour l'inférence : une liste de textes suffit.
Un dataset devient utile pour l'évaluation avec labels réels ou graphique temporel.

## Financial PhraseBank — recommandé pour évaluation

[Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
contient des phrases de news financières anglaises annotées `negative`, `neutral` ou
`positive`.

Configurations disponibles :

| Fichier | Accord annotateurs | Lignes |
|---|---:|---:|
| `Sentences_AllAgree.txt` | 100 % | 2 264 |
| `Sentences_75Agree.txt` | ≥ 75 % | 3 453 |
| `Sentences_66Agree.txt` | ≥ 66 % | 4 217 |
| `Sentences_50Agree.txt` | ≥ 50 % | 4 846 |

Commencer avec `AllAgree` : labels plus propres et volume réduit.

- [Télécharger archive ZIP](https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip)
- [Article associé](https://arxiv.org/abs/1307.5336)
- Licence : CC BY-NC-SA 3.0. Usage commercial interdit sans autorisation.

Le Dataset Viewer Hugging Face ne charge actuellement pas cet ancien dataset scripté.
L'archive ZIP directe reste le chemin fiable.

### Lecture

Après extraction :

```python
from pathlib import Path

import pandas as pd

path = Path("data/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt")
rows = []

for line in path.read_text(encoding="iso-8859-1").splitlines():
    text, label = line.rsplit("@", 1)
    rows.append({"text": text, "label": label})

dataset = pd.DataFrame(rows)
```

Utiliser `dataset["text"]` avec `predict()` et `dataset["label"]` avec `evaluate()`.

## FNSPID — news datées

[FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) fournit notamment date,
titre et symbole boursier. Adapté à `plot_timeline()`.

Points importants :

- plus de 10 millions de news ;
- environ 29,6 Go ;
- le Dataset Viewer rencontre des erreurs de génération ;
- licence CC BY-NC 4.0, donc usage commercial interdit sans autorisation.

Ce dataset est trop gros pour un premier exemple. Préférer un petit CSV personnel avec
les colonnes :

```csv
text,published_at
The company raised its guidance,2026-08-01
Revenue remained unchanged,2026-08-02
Quarterly earnings declined sharply,2026-08-03
```

Puis :

```python
import pandas as pd

news = pd.read_csv("news.csv")
predictions = analyzer.predict(news["text"])
predictions["published_at"] = news["published_at"]
analyzer.plot_timeline(predictions, date_col="published_at", freq="D")
```

## Choix rapide

- Tester inférence : trois phrases écrites à main.
- Tester évaluation + confusion matrix : Financial PhraseBank `AllAgree`.
- Tester timeline : petit CSV daté.
- Étudier millions de news : FNSPID seulement après MVP validé.
