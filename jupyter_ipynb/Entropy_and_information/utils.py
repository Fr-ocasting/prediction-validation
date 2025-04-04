import numpy as np 
def correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcule le coefficient de corrélation de Pearson entre x et y.
    """
    if len(x) < 2:
        return 0.0
    return np.corrcoef(x, y)[0, 1]