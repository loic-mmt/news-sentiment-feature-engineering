# Plan — mini-librairie d'analyse de sentiment financier

## Versions

- **v1.0 — MVP actuel :** inférence, résultats, statistiques, évaluation et graphiques.
- **v1.1 — prochaine étape :** fine-tuning, collecte de news et création de features
  point-in-time pour la recherche de signaux de trading.

La v1.1 doit rester une librairie locale et légère. Elle ne devient ni une plateforme
de données, ni un moteur de backtest, ni un service temps réel administré.

## 1. Objectif

Transformer les éléments utiles de `course/` en une petite librairie Python capable de :

1. prédire le sentiment d'un texte ou d'un lot de textes ;
2. retourner le label, la confiance, les probabilités par classe et un score continu ;
3. calculer quelques statistiques descriptives ;
4. évaluer les prédictions lorsque les vrais labels sont disponibles ;
5. produire quelques graphiques simples, sans imposer leur affichage ni leur sauvegarde.

Le cas d'usage principal est l'analyse de titres et de courts textes financiers en anglais.

## 2. Périmètre de la v1.0

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

Le fine-tuning, la collecte de news et la création de features passent dans la v1.1.
Les autres éléments ne seront ajoutés que si un besoin concret apparaît.

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

## 9. Objectif de la v1.1

Fournir une chaîne courte et reproductible :

```text
news horodatées → nettoyage/déduplication → sentiment → agrégation temporelle
                → features par ticker → évaluation face aux rendements futurs
```

La librairie crée des features. Elle ne prend aucune décision d'achat ou de vente.

Trois capacités sont ajoutées :

1. fine-tuner un checkpoint sur un dataset `text`/`label` ;
2. récupérer régulièrement des news financières textuelles ;
3. produire des features sans fuite temporelle, directement joignables à des données
   de marché.

## 10. Principes de conception de la v1.1

- Conserver l'API d'inférence existante et charger un checkpoint fine-tuné avec
  `SentimentAnalyzer(model_name=chemin_du_checkpoint)`.
- Séparer les dépendances optionnelles : `.[train]` pour l'entraînement et `.[news]`
  pour l'acquisition. L'installation de base reste légère.
- Utiliser des fonctions et des DataFrames plutôt qu'une hiérarchie de classes.
- Stocker localement en Parquet ; aucune base de données en v1.1.
- Privilégier une API ou un flux RSS qui fournit légalement le texte. Le scraping HTML
  n'est utilisé que pour une source explicitement autorisée, sans contourner paywall,
  robots.txt ou conditions d'utilisation.
- Exécuter la collecte via un script idempotent appelable par `cron`. Aucun daemon,
  queue ou orchestrateur interne.
- Utiliser d'abord les probabilités et agrégats simples. Les embeddings et modèles
  supplémentaires ne sont ajoutés que si une évaluation hors échantillon justifie
  leur coût.

## 11. Contrats de données

### Dataset annoté pour le fine-tuning

Colonnes minimales :

| Colonne | Contenu |
|---|---|
| `text` | Texte financier non vide |
| `label` | `negative`, `neutral` ou `positive` |

Colonnes recommandées : `published_at`, `source`, `ticker` et `document_id`. Elles
permettent un split temporel ou par source et empêchent qu'un doublon apparaisse dans
plusieurs partitions.

### News normalisées

| Colonne | Contenu |
|---|---|
| `news_id` | Identifiant stable ou hash déterministe |
| `published_at` | Date annoncée par la source, en UTC |
| `available_at` | Première date à laquelle le pipeline a observé la news, en UTC |
| `source` | Fournisseur ou flux |
| `url` | URL canonique si disponible |
| `ticker` | Symbole fourni par la source ou résolu par une table d'alias explicite |
| `title` | Titre de la news |
| `text` | Résumé ou corps utilisable par le modèle |

`available_at` est le timestamp de référence pour les features point-in-time. Utiliser
uniquement `published_at` pourrait introduire une news avant son arrivée réelle dans
le pipeline.

Le mapping vers les tickers reste volontairement simple : métadonnée du fournisseur,
puis table d'alias fournie par l'utilisateur. Le NER et l'entity linking restent hors
v1.1.

## 12. Fine-tuning minimal

Ajouter `src/news_sentiment/training.py` avec une fonction publique unique :

```python
run = fine_tune(
    data,
    output_dir="artifacts/finbert-custom",
    validation_data=validation_data,
    test_data=test_data,
    base_model="yiyanghkust/finbert-tone",
)

analyzer = SentimentAnalyzer(model_name=run.best_model_path)
```

La fonction doit :

- accepter un DataFrame ou un chemin CSV/Parquet ;
- valider et normaliser les trois labels ;
- dédupliquer avant de découper les données ;
- créer des splits stratifiés si validation et test ne sont pas fournis ;
- préférer un split temporel ou par source lorsque les colonnes existent ;
- utiliser le protocole de départ décrit dans [MODEL_NOTES.md](MODEL_NOTES.md) ;
- sélectionner le checkpoint sur la macro-F1 de validation avec early stopping ;
- mesurer aussi la log-loss, car les probabilités deviennent des features ;
- sauvegarder tokenizer, modèle, mappings de labels, hyperparamètres et métriques ;
- retourner un petit `TrainingRun`, sans exposer directement les objets du Trainer.

Un seul modèle est entraîné par appel. Pas de recherche distribuée d'hyperparamètres.
Un petit essai borné sur le learning rate ou le nombre d'époques pourra être ajouté
seulement si la baseline montre un gain stable sur validation externe.

Le test final ne sert jamais à choisir les hyperparamètres. Attention particulière à
la fuite liée aux checkpoints déjà entraînés sur Financial PhraseBank.

## 13. Acquisition de news

Ajouter `src/news_sentiment/news.py` avec un premier connecteur générique RSS :

```python
news = fetch_rss(feed_urls, since=last_run_at, ticker_aliases=aliases)
news = normalize_news(news)
news = deduplicate_news(news)
save_news(news, "data/news.parquet")
```

Ordre d'implémentation :

1. RSS contenant titre et résumé ;
2. une API financière seulement si elle apporte de meilleurs tickers, texte ou
   horodatages ;
3. extraction de pages HTML uniquement pour une source autorisée et stable.

La déduplication v1.1 utilise l'URL canonique, puis un hash du titre et du texte
normalisés. Elle doit fonctionner entre deux exécutions, pas seulement dans le lot
courant.

La collecte doit être incrémentale et idempotente : relancer le script avec la même
fenêtre ne crée aucune nouvelle ligne. Les erreurs d'un flux sont remontées sans faire
perdre les news déjà récupérées auprès des autres flux.

## 14. Features de sentiment pour le trading

Ajouter `src/news_sentiment/features.py` :

```python
scored_news = attach_sentiment(news, analyzer, text_col="text")

features = build_sentiment_features(
    scored_news,
    timestamp_col="available_at",
    group_col="ticker",
    frequency="1D",
    lookback="24h",
    half_life="6h",
)
```

`attach_sentiment()` appelle l'analyseur puis rattache les probabilités et le score
aux lignes originales sans perdre `news_id`, `ticker` ou les timestamps. Elle vérifie
l'alignement et conserve l'ordre ; elle ne fait aucun calcul supplémentaire.

Pour chaque ticker et instant de décision, produire uniquement :

- `news_count` : nombre de news uniques ;
- `sentiment_mean` et `sentiment_std` ;
- `p_positive_mean`, `p_neutral_mean`, `p_negative_mean` ;
- `positive_share` et `negative_share` ;
- `confidence_mean` ;
- `sentiment_ewm` : moyenne pondérée par récence et confiance ;
- `sentiment_momentum` : différence entre fenêtre courte et fenêtre longue ;
- `hours_since_last_news`.

Toutes les fenêtres sont strictement rétrospectives. Une feature horodatée à `t` ne
peut utiliser que les news dont `available_at <= t`. Une ligne sans news conserve
`news_count = 0` et des valeurs manquantes explicites pour les mesures non définies ;
le remplissage appartient ensuite au modèle de trading.

Les labels seuls ne doivent pas remplacer les probabilités continues. Les embeddings,
la « news surprise », les poids appris et les dizaines d'indicateurs dérivés restent
hors v1.1 : le cours ne montre pas de gain suffisamment robuste pour justifier leur
complexité.

## 15. Évaluation des features

Ajouter une fonction légère `evaluate_features()` dans `features.py`. Elle aligne les
features à des rendements futurs déjà fournis par l'utilisateur et calcule :

- couverture par date et ticker ;
- corrélation de Spearman cross-sectionnelle quotidienne (IC) ;
- moyenne, écart-type et ICIR de l'IC ;
- spread de rendement entre quintiles extrêmes ;
- résultats par horizon futur demandé.

Cette fonction ne télécharge pas les prix et ne simule ni ordres, ni coûts, ni
portefeuille. Elle sert à éliminer rapidement les features inutiles. La validation
finale doit être temporelle ou walk-forward, jamais un split aléatoire des lignes.

Le modèle de sentiment est sélectionné sur ses métriques de classification et de
probabilités. Les métriques de rendement servent ensuite à valider la feature, sans
réutiliser la période de test pour optimiser le modèle.

## 16. Structure cible de la v1.1

```text
news-sentiment/
├── src/news_sentiment/
│   ├── analyzer.py          # inférence, inchangée dans son principe
│   ├── training.py          # fine_tune() et TrainingRun
│   ├── news.py              # RSS, normalisation, déduplication, persistance
│   └── features.py          # enrichissement, agrégats et évaluation rapide
├── examples/
│   ├── fine_tune.py
│   └── realtime_features.py
├── data/                    # ignoré par Git, aucune donnée redistribuée
└── artifacts/               # ignoré par Git, checkpoints locaux
```

Pas de module supplémentaire tant que ces trois nouveaux fichiers restent lisibles.

## 17. Ordre d'implémentation de la v1.1

### Étape 1 — Fine-tuning reproductible

Implémenter `fine_tune()` et recharger son meilleur checkpoint avec l'analyseur
existant. Valider d'abord sur un petit dataset local.

**Terminé lorsque :** une commande entraîne, sauvegarde, recharge puis prédit sans
notebook ni modification manuelle des labels.

### Étape 2 — Collecte incrémentale

Implémenter un flux RSS bout en bout, normalisation UTC, mapping simple des tickers,
déduplication et stockage Parquet.

**Terminé lorsque :** deux exécutions identiques produisent les mêmes lignes et une
nouvelle exécution n'ajoute que les news réellement nouvelles.

### Étape 3 — Feature frame point-in-time

Enrichir les news avec `SentimentAnalyzer.predict()`, puis calculer les agrégats par
ticker et fenêtre.

**Terminé lorsque :** déplacer artificiellement une news dans le futur ne modifie
aucune feature antérieure.

### Étape 4 — Validation face aux rendements

Implémenter les IC, ICIR et quintiles avec des rendements futurs fournis par
l'utilisateur. Comparer au minimum le modèle par défaut et le checkpoint fine-tuné.

**Terminé lorsque :** le même découpage temporel produit un rapport reproductible et
qu'aucun rendement futur n'entre dans le calcul des features.

### Étape 5 — Exemple exécutable

Créer un script court : récupération des nouvelles observations, prédiction,
persistance et export des features du jour. Documenter son lancement périodique par
`cron` sans imposer `cron` à la librairie.

**Terminé lorsque :** un utilisateur peut générer un fichier de features actualisé
avec une seule commande après configuration de ses flux.

## 18. Hors périmètre de la v1.1

- moteur de backtest, exécution d'ordres ou recommandations d'investissement ;
- websocket, streaming milliseconde ou garantie de disponibilité ;
- base SQL, orchestration distribuée et dashboard temps réel ;
- scraping générique du Web ou contournement de paywalls ;
- résolution automatique complexe des entreprises et tickers ;
- entraînement multi-GPU, AutoML ou registre de modèles ;
- embeddings, topic modeling, résumé automatique et NER ;
- support multilingue.

## 19. Critères d'acceptation de la v1.1

- Un dataset `text`/`label` peut produire un checkpoint rechargeable par
  `SentimentAnalyzer`.
- Le rapport d'entraînement conserve la macro-F1, la log-loss, les paramètres et la
  provenance des données.
- Une source RSS est collectée de manière incrémentale avec des timestamps UTC et sans
  doublons exacts.
- Le schéma des news distingue publication et disponibilité réelle.
- Les features sont calculées par ticker avec des fenêtres strictement rétrospectives.
- Un rapport IC/ICIR/quintiles peut être construit avec des rendements fournis.
- Un exemple relie collecte, sentiment et export de features sans serveur ni notebook.
- La v1.0 reste utilisable sans installer les dépendances `train` ou `news`.
