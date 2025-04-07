import numpy as np 
def correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcule le coefficient de corrélation de Pearson entre x et y.
    """
    if len(x) < 2:
        return 0.0
    return np.corrcoef(x, y)[0, 1]



def get_minmax(te_xy):
    """
    Fonction pour récupérer le maximum d'une série de TE.
    """
    max_te = max(te_xy)
    min_te = min(te_xy)
    if min_te < -1e-5:
       return min_te,max_te
    elif max_te < 1e-5:
        return 0,1.0
    else:
        return min_te,max_te