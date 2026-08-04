# Plan — mini-librairie d'analyse de sentiment financier

## 1. Objectif

Transformer les éléments utiles de `course/` en une petite librairie Python capable de :

1. prédire le sentiment d'un texte ou d'un lot de textes ;
2. retourner le label, la confiance, les probabilités par classe et un score continu ;
3. calculer quelques statistiques descriptives ;
4. évaluer les prédictions lorsque les vrais labels sont disponibles ;
5. produire quelques graphiques simples, sans imposer leur affichage ni leur sauvegarde.

Le cas d'usage principal est l'analyse de titres et de courts textes financiers en anglais.

## 2. Périmètre du MVP

### Inclus

- Modèle par défaut : `yiyanghkust/finbert-tone`, déjà employé dans
  `course/03_sentiment_evolution.py` et `course/07_news_return_signals.py`.
- Inférence sur `str` et `Sequence[str]`, avec traitement par lots.
- Détection automatique de CPU/GPU, avec possibilité de forcer le device.
- Labels normalisés : `negative`, `neutral`, `positive`.
- Sortie sous forme de `pandas.DataFrame` avec une ligne par texte :
  - `text`
  - `label`
  - `confidence` (probabilité du label prédit)
  - `p_negative`
  - `p_neutral`
  - `p_positive`
  - `sentiment_score = p_positive - p_negative`, donc compris entre -1 et 1
- Statistiques descriptives : effectifs/proportions par label, score moyen et médian,
  confiance moyenne et taux de prédictions sous un seuil de confiance.
- Évaluation optionnelle avec labels réels : accuracy, macro-F1, rapport par classe et
  matrice de confusion.
- Graphiques : distribution des labels, distribution du score continu, évolution
  temporelle du score si une colonne de dates est fournie, et matrice de confusion.

### Hors périmètre

- Fine-tuning ou entraînement de modèles.
- Comparaison automatique entre TF-IDF, GloVe et Transformers.
- NER, embeddings, déduplication de news et signaux de rendement des scripts `05`,
  `07`, `08` et `09`.
- API HTTP, interface web, base de données, CLI élaborée ou système de plugins.
- Explicabilité SHAP, gestion multilingue et calibration avancée des probabilités.

Ces éléments ne seront ajoutés que si un besoin concret apparaît après le MVP.

## 3. API publique proposée

```python
from news_sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Un texte : résultat Python simple
result = analyzer.predict_one("The company raised its annual guidance.")
# Prediction(label="positive", confidence=..., probabilities=..., score=...)

# Plusieurs textes : DataFrame directement exploitable
predictions = analyzer.predict(
    [
        "The company raised its annual guidance.",
        "Revenue was unchanged from last year.",
    ]
)

summary = analyzer.summarize(predictions, confidence_threshold=0.60)
ax = analyzer.plot_labels(predictions)
```

Évaluation avec des labels réels :

```python
report = analyzer.evaluate(
    predictions,
    y_true=["positive", "neutral"],
)

ax = analyzer.plot_confusion_matrix(report)
```

Évolution temporelle :

```python
predictions["published_at"] = dates
ax = analyzer.plot_timeline(predictions, date_col="published_at", freq="D")
```

Principes d'API :

- le modèle et le tokenizer sont chargés une seule fois, à la création de l'analyseur ;
- les fonctions de tracé retournent un `matplotlib.axes.Axes` et n'appellent jamais
  `plt.show()` ;
- aucune fonction ne sauvegarde de fichier sans demande explicite de l'appelant ;
- les entrées vides ou non textuelles produisent une erreur claire ;
- l'ordre des résultats reste identique à celui des textes d'entrée.

## 4. Structure cible

```text
news-sentiment/
├── pyproject.toml
├── README.md
├── src/
│   └── news_sentiment/
│       ├── __init__.py       # exports publics uniquement
│       ├── analyzer.py       # chargement du modèle et inférence
│       ├── results.py        # Prediction et EvaluationReport
│       ├── stats.py          # résumés et métriques
│       └── plots.py          # visualisations Matplotlib
└── tests/
    ├── test_analyzer.py
    ├── test_stats.py
    └── test_plots.py
```

Éviter un découpage plus fin tant que ces modules restent courts.

## 5. Étapes d'implémentation

### Étape 1 — Socle installable

- Créer `pyproject.toml` avec Python 3.11+ et les dépendances strictement utiles :
  `transformers`, `torch`, `pandas`, `numpy`, `scikit-learn` et `matplotlib`.
- Créer le package `src/news_sentiment` et définir ses exports publics.
- Ajouter dans le README l'installation et un exemple minimal exécutable.

**Terminé lorsque :** `pip install -e .` fonctionne et
`from news_sentiment import SentimentAnalyzer` est valide.

### Étape 2 — Inférence et contrat de sortie

- Implémenter `Prediction` avec : `label`, `confidence`, `probabilities` et `score`.
- Charger le tokenizer et le modèle Hugging Face dans `SentimentAnalyzer`.
- Demander les scores de toutes les classes, puis normaliser les noms de labels à
  partir de la configuration du modèle plutôt que de dépendre de leur ordre.
- Implémenter `predict_one()` et `predict()` avec `batch_size`, troncature et
  `max_length` configurables.
- Garantir les invariants : probabilités dans `[0, 1]`, somme proche de 1,
  `confidence == max(probabilities)` et score dans `[-1, 1]`.

**Terminé lorsque :** un lot conserve son ordre et produit exactement les colonnes
documentées sur CPU comme sur GPU.

### Étape 3 — Statistiques et évaluation

- Implémenter `summarize()` sur le DataFrame de prédictions.
- Implémenter `evaluate()` en validant les labels réels et leur longueur.
- Retourner un `EvaluationReport` contenant les métriques scalaires, le rapport par
  classe et la matrice de confusion ; ne pas mélanger calcul et affichage.

**Terminé lorsque :** les résultats sont comparés sur de petits cas déterministes à
ceux de scikit-learn.

### Étape 4 — Graphiques essentiels

- `plot_labels()` : barres des effectifs ou proportions.
- `plot_score_distribution()` : histogramme du `sentiment_score` avec zéro visible.
- `plot_timeline()` : moyenne du score par période, après conversion et tri des dates.
- `plot_confusion_matrix()` : heatmap simple à partir du rapport d'évaluation.
- Accepter un axe existant via `ax=None` pour permettre la composition de figures.

**Terminé lorsque :** chaque fonction retourne un `Axes`, fonctionne avec le backend
non interactif `Agg` et ne crée aucun fichier par effet de bord.

### Étape 5 — Documentation et validation finale

- Documenter les formats d'entrée/sortie, le choix du modèle et la signification du
  score continu.
- Ajouter un exemple complet : CSV de news → prédictions → résumé → graphique.
- Lancer les tests unitaires et un smoke test réel sur trois phrases, marqué comme
  test lent afin que les tests ordinaires ne téléchargent pas le modèle.

**Terminé lorsque :** une nouvelle personne peut installer le projet et reproduire
l'exemple du README sans lire le code de `course/`.

## 6. Stratégie de test minimale

- Simuler le pipeline Hugging Face dans les tests unitaires : pas de réseau ni de
  téléchargement du modèle dans la suite standard.
- Tester un texte, un lot, un générateur, l'entrée vide et un type invalide.
- Tester la normalisation de différentes casses de labels (`Positive`, `positive`).
- Vérifier les colonnes, l'ordre, les bornes et la somme des probabilités.
- Tester les statistiques sur un jeu construit à la main.
- Tester les erreurs de labels réels invalides ou de longueur différente.
- Vérifier seulement le type, les titres et les données principales des graphiques,
  sans tests d'images fragiles.
- Conserver un unique smoke test d'intégration avec le vrai checkpoint.

## 7. Décisions à conserver simples

- Un seul backend de modèle dans le MVP ; `model_name` reste configurable pour ne pas
  bloquer un changement futur.
- `pandas` est le format public ; le support Polars peut être ajouté plus tard si un
  usage réel le justifie.
- Le score continu est transparent et reproductible (`P(positive) - P(negative)`),
  plutôt qu'une nouvelle heuristique.
- Pas de classe de configuration dédiée : quelques arguments explicites sur
  `SentimentAnalyzer` suffisent.
- Pas de cache applicatif ; Hugging Face gère déjà le cache local des modèles.

## 8. Critères d'acceptation du MVP

- L'import public tient dans `from news_sentiment import SentimentAnalyzer, Prediction`.
- Une seule initialisation permet plusieurs appels d'inférence sans recharger le modèle.
- Chaque prédiction expose les trois probabilités, le label, la confiance et le score.
- Les statistiques fonctionnent directement sur la sortie de `predict()`.
- Les quatre graphiques retournent des objets Matplotlib réutilisables.
- Les tests unitaires standards passent hors ligne.
- Le README contient un parcours utilisable en moins de dix lignes de code.
- Aucun code d'entraînement, serveur ou abstraction non nécessaire n'entre dans le MVP.
