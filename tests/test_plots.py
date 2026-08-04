"""TODO checklist for plots.py."""

# TODO 1: forcer le backend Matplotlib Agg dans la configuration des tests, pas
#         dans le module de production.
# TODO 2: vérifier que _get_ax réutilise l'axe fourni et crée un Axes sinon.
# TODO 3: vérifier que plot_labels crée toujours trois barres dans l'ordre stable,
#         y compris lorsqu'une classe est absente.
# TODO 4: vérifier counts et proportions séparément.
# TODO 5: vérifier le nombre de bins, xlim=(-1, 1) et la ligne verticale à zéro
#         de plot_score_distribution.
# TODO 6: vérifier l'agrégation journalière et hebdomadaire de plot_timeline ainsi
#         que les erreurs de colonne/date/fréquence.
# TODO 7: vérifier les modes brut et normalisé de plot_confusion_matrix, y compris
#         une ligne dont la somme vaut zéro.
# TODO 8: vérifier que toutes les fonctions retournent un Axes, ne mutent pas les
#         entrées et ne créent aucun fichier ; éviter les comparaisons pixel à pixel.
