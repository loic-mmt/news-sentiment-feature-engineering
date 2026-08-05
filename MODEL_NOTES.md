# Notes de modélisation

Ce document conserve les enseignements durables des expériences présentes dans
`course/`. Il ne décrit pas une promesse de performance et ne remplace pas une
évaluation sur les données réellement visées par le projet.

## Décision actuelle

La mini-librairie utilise par défaut
[`yiyanghkust/finbert-tone`](https://huggingface.co/yiyanghkust/finbert-tone) pour
l'inférence sur des textes financiers anglais.

Le fine-tuning reste hors de la v1.0 : prédiction, statistiques, évaluation et
graphiques sont prioritaires. Il est prévu pour la v1.1 dans un module optionnel
`src/news_sentiment/training.py`, afin que l'installation de base reste légère.

## Quand envisager un fine-tuning

Un fine-tuning devient pertinent lorsque :

- des exemples annotés et représentatifs de la source cible sont disponibles ;
- une évaluation externe montre que le checkpoint actuel généralise mal ;
- la définition métier des labels est claire et stable ;
- une baseline simple et reproductible existe déjà.

Il ne faut pas fine-tuner uniquement parce qu'un modèle plus complexe semble pouvoir
améliorer son score sur un jeu de données déjà connu.

## Baselines et métriques

Une régression logistique sur des représentations TF-IDF constitue une baseline utile.
Dans les expériences du cours, elle atteint déjà environ 83 % d'accuracy sur une
version très consensuelle de Financial PhraseBank. Un Transformer doit donc être
comparé à cette baseline, ainsi qu'à une prédiction de la classe majoritaire.

L'accuracy seule est insuffisante. Toute évaluation doit au minimum conserver :

- la macro-F1 ;
- les scores par classe ;
- la matrice de confusion ;
- la distribution des classes ;
- les résultats sur une source externe ou une période différente.

Voir [`course/03_sentiment_evolution.py`](course/03_sentiment_evolution.py).

## Protocole de fine-tuning de référence

La configuration suivante constitue un point de départ, pas un réglage universel :

```text
learning_rate = 2e-5
batch_size = 16
epochs = 3
weight_decay = 0.01
warmup_ratio = 0.1
max_length = 128
early_stopping_patience = 2
```

Le protocole doit également comprendre :

- un split stratifié train/validation/test ;
- un padding dynamique ;
- des mappings `id2label` et `label2id` explicites ;
- une sélection du meilleur checkpoint sur la validation ;
- un arrêt anticipé ;
- une graine aléatoire et les versions des dépendances enregistrées ;
- un test final exécuté une seule fois après choix des hyperparamètres.

Voir [`course/04_bert_finetuning.py`](course/04_bert_finetuning.py).

## Fuite de données et comparaisons de modèles

`ProsusAI/finbert` a déjà été entraîné sur Financial PhraseBank. Le fine-tuner puis
l'évaluer sur un nouveau split aléatoire de ce même corpus peut contaminer le test et
surestimer la généralisation. Les très hauts scores obtenus ainsi ne constituent pas
une preuve suffisante pour un déploiement.

`ProsusAI/finbert` et `yiyanghkust/finbert-tone` sont deux checkpoints distincts,
entraînés avec des données et objectifs différents. Leurs résultats ne doivent pas
être comparés comme s'il s'agissait du même modèle avant et après fine-tuning.

De même, l'utilisation de `finbert-tone` sans nouvel entraînement est un transfert
entre jeux de données, mais pas du « zero-shot » au sens strict : ce checkpoint possède
déjà une tête de classification de sentiment entraînée.

## Distribution shift

Des labels portant les mêmes noms ne garantissent pas une tâche identique. Financial
PhraseBank représente un jugement humain sur une phrase financière, tandis que des
datasets comme FinMarBa peuvent dériver les labels de la réaction du marché.

Avant tout déploiement, il faut donc tester le modèle sur :

- les sources de nouvelles réellement utilisées ;
- des périodes non présentes dans les données d'entraînement ;
- des secteurs et sociétés variés ;
- la définition métier exacte de `negative`, `neutral` et `positive`.

Une baisse de performance entre datasets indique souvent un changement de domaine ou
d'objectif, pas nécessairement une erreur de mapping des labels.

Voir [`course/06_finbert_cross_dataset.py`](course/06_finbert_cross_dataset.py).

## Documents longs

Les modèles de type BERT acceptent une longueur limitée. Tronquer systématiquement un
rapport long peut supprimer l'information importante. Pour les rapports ou filings :

1. découper le document en phrases ou chunks ;
2. prédire chaque chunk ;
3. agréger le score moyen, sa dispersion et les proportions de labels ;
4. conserver la date de publication réelle pour toute analyse temporelle.

Voir [`course/09_filing_text_signals.py`](course/09_filing_text_signals.py).

## Sentiment et signal de marché

Un bon classifieur de sentiment ne produit pas automatiquement un signal de trading.
Les expériences du cours obtiennent des IC et ICIR très faibles, parfois avec une
relation économique dans le mauvais sens.

Toute expérimentation de signal doit respecter les règles suivantes :

- une nouvelle publiée au jour `t` ne peut expliquer qu'un rendement futur ;
- les statistiques historiques ne doivent utiliser aucune observation future ;
- les dépêches dupliquées ou syndiquées doivent être dédupliquées ;
- les tickers et dates doivent être normalisés ;
- la couverture, la dispersion et le momentum peuvent compléter la moyenne du
  sentiment ;
- la monotonie des portefeuilles triés doit être contrôlée, pas seulement la
  significativité statistique.

Voir [`course/07_news_return_signals.py`](course/07_news_return_signals.py) et
[`course/08_text_feature_evaluation.py`](course/08_text_feature_evaluation.py).

## NER : extension possible

La reconnaissance d'entités nommées pourrait plus tard associer le sentiment aux
entreprises, montants, dates et pourcentages cités. Elle n'est pas nécessaire au MVP.

Pour un futur fine-tuning NER, l'alignement entre tokens et sous-tokens ainsi que les
labels BIO sont essentiels. Des scores parfaits sur des données synthétiques ne
démontrent pas une généralisation sur de vraies nouvelles financières.

Voir [`course/05_financial_ner_finetuning.py`](course/05_financial_ner_finetuning.py).

## Informations à enregistrer pour chaque expérience

Pour rendre un résultat reproductible, conserver au minimum :

- le nom et la révision du checkpoint ;
- la provenance, la version et la licence du dataset ;
- la définition et la distribution des labels ;
- la méthode de découpage des données ;
- les hyperparamètres et la graine ;
- les métriques globales et par classe ;
- les limites connues et les risques de fuite ;
- la période et les sources couvertes par le test externe.
