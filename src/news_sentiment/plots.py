"""Small, composable Matplotlib visualizations."""

from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes

from .results import SENTIMENT_LABELS, EvaluationReport
from .stats import validate_prediction_frame

# Palette stable dans toute la librairie : rouge, gris, vert.
SENTIMENT_COLORS: dict[str, str] = {
    "negative": "#C44E52",
    "neutral": "#8C8C8C",
    "positive": "#55A868",
}

# Ordre conseillé : _get_ax -> plot_labels -> plot_score_distribution ->
# plot_confusion_matrix -> plot_timeline. Aucun tracé ne doit appeler show/savefig.


def _get_ax(ax: Axes | None, *, figsize: tuple[float, float]) -> Axes:
    """Return the supplied axis or create a new one."""
    # TODO 1: si ax n'est pas None, vérifier isinstance(ax, Axes) puis le retourner.
    # TODO 2: sinon importer matplotlib.pyplot localement.
    # TODO 3: créer _, ax = plt.subplots(figsize=figsize) puis retourner ax.
    # TODO 4: ne jamais changer le backend Matplotlib dans le code de la librairie.
    raise NotImplementedError


def plot_labels(
    predictions: pd.DataFrame,
    *,
    proportions: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot label counts, or proportions, in stable sentiment order."""
    # TODO 1: appeler validate_prediction_frame(predictions).
    # TODO 2: calculer les effectifs et reindexer avec SENTIMENT_LABELS pour
    #         afficher également les classes absentes avec une barre à zéro.
    # TODO 3: si proportions=True, diviser par len(predictions).
    # TODO 4: récupérer ax avec _get_ax(ax, figsize=(7, 4)).
    # TODO 5: tracer une barre par classe avec SENTIMENT_COLORS dans l'ordre stable.
    # TODO 6: définir titre, xlabel="Sentiment" et ylabel selon counts/proportions.
    # TODO 7: optionnel MVP utile : annoter chaque barre avec sa valeur lisible.
    # TODO 8: retourner ax sans appeler tight_layout, show ou savefig ; l'appelant
    #         conserve le contrôle de la figure complète.
    raise NotImplementedError


def plot_score_distribution(
    predictions: pd.DataFrame,
    *,
    bins: int = 30,
    ax: Axes | None = None,
) -> Axes:
    """Plot a histogram of ``sentiment_score`` with a zero reference line."""
    # TODO 1: appeler validate_prediction_frame(predictions).
    # TODO 2: rejeter bool et valider bins comme entier >= 1.
    # TODO 3: récupérer ax avec _get_ax(ax, figsize=(8, 4)).
    # TODO 4: tracer ax.hist(sentiment_score, bins=bins, range=(-1, 1), ...).
    # TODO 5: tracer ax.axvline(0, ...) pour séparer polarité négative/positive.
    # TODO 6: fixer xlim(-1, 1), titre et noms des axes.
    # TODO 7: retourner ax sans afficher ni enregistrer la figure.
    raise NotImplementedError


def plot_timeline(
    predictions: pd.DataFrame,
    *,
    date_col: str,
    freq: str = "D",
    ax: Axes | None = None,
) -> Axes:
    """Plot mean sentiment score resampled over time."""
    # TODO 1: appeler validate_prediction_frame(predictions).
    # TODO 2: vérifier que date_col est une chaîne non vide et existe dans le frame.
    # TODO 3: travailler sur predictions[[date_col, "sentiment_score"]].copy()
    #         pour ne jamais modifier les données de l'appelant.
    # TODO 4: convertir avec pd.to_datetime(..., errors="coerce", utc=True) ;
    #         lever ValueError en indiquant le nombre de dates non convertibles.
    # TODO 5: valider freq en tentant le resample et transformer l'erreur Pandas en
    #         ValueError lisible (exemples documentés : D, W, ME).
    # TODO 6: trier, placer les dates en index, puis calculer la moyenne du score
    #         par période ; supprimer seulement les périodes sans observation.
    # TODO 7: récupérer ax avec _get_ax(ax, figsize=(10, 4)), tracer la série et
    #         ajouter une ligne horizontale à zéro.
    # TODO 8: définir titre, xlabel="Date", ylabel="Mean sentiment score".
    # TODO 9: retourner ax ; ne pas appeler plt.show() ni modifier predictions.
    raise NotImplementedError


def plot_confusion_matrix(
    report: EvaluationReport,
    *,
    normalize: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot raw counts or row-normalized values from an evaluation report."""
    # TODO 1: vérifier isinstance(report, EvaluationReport).
    # TODO 2: convertir report.confusion_matrix en array numérique sans modifier
    #         l'original, puis vérifier shape == (len(report.labels),) * 2.
    # TODO 3: rejeter NaN, infinis et valeurs négatives.
    # TODO 4: si normalize=True, diviser chaque ligne par sa somme avec np.divide
    #         et where=row_sum != 0 pour gérer une vraie classe absente.
    # TODO 5: récupérer ax avec _get_ax(ax, figsize=(6, 5)).
    # TODO 6: dessiner avec ax.imshow ; ajouter une colorbar via ax.figure.colorbar.
    # TODO 7: poser les ticks/labels dans report.labels, avec axe x=Predicted et
    #         axe y=True.
    # TODO 8: annoter chaque cellule en entier ou en .2f selon normalize ; choisir
    #         une couleur de texte lisible selon la valeur de la cellule.
    # TODO 9: retourner ax sans show/savefig. Voir les matrices du script
    #         course/03_sentiment_evolution.py pour le résultat visuel attendu.
    raise NotImplementedError
