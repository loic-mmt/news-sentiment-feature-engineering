"""TODO checklist for stats.py."""

# TODO 1: créer une fixture DataFrame minimale avec une ligne par sentiment et
#         des probabilités calculées à la main.
# TODO 2: vérifier chaque règle de validate_prediction_frame indépendamment :
#         colonne absente, frame vide, null, label invalide, NaN/inf, bornes,
#         somme des probabilités, confidence, label gagnant et score continu.
# TODO 3: confirmer que les colonnes supplémentaires sont acceptées et que le
#         DataFrame d'entrée n'est jamais modifié.
# TODO 4: vérifier summarize sur un lot déséquilibré et la présence d'une classe
#         absente avec count=0/proportion=0.
# TODO 5: tester confidence_threshold à 0, 1, hors bornes, NaN et bool.
# TODO 6: comparer accuracy, macro-F1 et confusion_matrix aux fonctions sklearn
#         sur un exemple déterministe.
# TODO 7: vérifier que evaluate retourne toujours une matrice 3x3 et trois lignes
#         per_class, même lorsqu'une classe est absente.
# TODO 8: tester y_true de mauvaise longueur, chaîne nue, label inconnu et casse
#         différente ; vérifier que predictions et y_true ne sont pas mutés.

