import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple
import itertools

import sys
import os

current_path = notebook_dir = os.getcwd()
working_dir = os.path.abspath(os.path.join(current_path, '..','..'))
if working_dir not in sys.path:
    sys.path.insert(0, working_dir)

from jupyter_ipynb.Entropy_and_information.utils import correlation_coefficient

def build_multivariate_embedding(
    data_dict: Dict[str, np.ndarray],
    lags_dict: Dict[str, List[int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit un "embedding" (attracteur multivarié) à partir
    de plusieurs séries et retards.
    
    Exemple:
      data_dict = {'x': x_array, 'z': z_array, ...}
      lags_dict = {'x': [0,1], 'z': [0], ...}
    
    => On empile [ x(t), x(t-1), z(t), ... ] dans un vecteur
       pour chaque t.
    
    Retourne:
      - embedding (shape (M, total_dims)) : le nuage de points dans R^{total_dims}
      - valid_time (shape (M,)) : l'indice temporel t associé à chaque ligne.
    """
    # On doit aligner tout le monde pour éviter de sortir de la série
    # => On calcule un max_lag par variable, puis un max global
    max_lag_global = 0
    for var, lags in lags_dict.items():
        max_lag_global = max(max_lag_global, max(lags)) if lags else max_lag_global

    # On suppose que toutes les séries ont la même longueur
    lengths = [len(arr) for arr in data_dict.values()]
    N = min(lengths)  # on va tronquer éventuellement

    # Nombre de "lignes" = N - max_lag_global
    M = N - max_lag_global
    if M <= 0:
        raise ValueError("Pas assez de points pour supporter le max lag global")

    # Construction
    # On crée d'abord une liste de colonnes => qu'on concaténera
    columns = []
    for var, series in data_dict.items():
        lags = lags_dict.get(var, [])
        for lag in lags:
            # On construit la colonne correspondante
            # i.e col[i] = series[i + max_lag_global - lag], index i=0..M-1
            col_data = []
            for i in range(M):
                t = i + max_lag_global  # indice réel dans la série
                col_data.append(series[t - lag])
            col_data = np.array(col_data)
            columns.append(col_data.reshape(-1,1))

    # On concatène horizontalement
    if len(columns) == 0:
        raise ValueError("Aucun lag spécifié => embedding vide")
    
    embedding = np.hstack(columns)  # shape (M, sum_of_all_lags)
    
    # valid_time[i] = i + max_lag_global
    valid_time = np.array([i + max_lag_global for i in range(M)], dtype=int)

    return embedding, valid_time

def ccm_reconstruct_multivariate(
    embedding: np.ndarray,
    valid_time: np.ndarray,
    effect_series: np.ndarray,
    k: int = 4,
    use_weighted_mean: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convergent Cross Mapping en mode multivarié :
    - embedding: shape (M, dim_embed), construit depuis [X,Z,...].
    - valid_time: array (M,) => l'indice temporel t pour chaque ligne.
    - effect_series: la série qu'on cherche à reconstruire (ex. Y).
    - k: nombre de plus proches voisins.
    - use_weighted_mean: si True, moyenne pondérée par 1/distance, sinon moyenne simple.
    
    Retourne (y_true, y_pred).
    """
    M = embedding.shape[0]
    # Fit NN sur l'embedding
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(embedding)

    y_true = []
    y_pred = []

    for i in range(M):
        # On cherche les k plus proches voisins de embedding[i]
        query_point = embedding[i].reshape(1, -1)
        distances, indices = nbrs.kneighbors(query_point)
        nn_ids = indices[0]   # shape (k,)

        # Récupérer Y(t_nb) pour t_nb = valid_time[nn_ids]
        neigh_times = valid_time[nn_ids]
        # Valeurs effectives
        val_neigh = effect_series[neigh_times]

        if use_weighted_mean:
            d = distances[0]  # shape (k,)
            d = np.where(d < 1e-12, 1e-12, d)  # éviter div 0
            w = 1.0 / d
            w /= w.sum()
            est_val = np.sum(val_neigh * w)
        else:
            # moyenne simple
            est_val = np.mean(val_neigh)

        # Valeur réelle
        t_i = valid_time[i]
        real_val = effect_series[t_i]

        y_true.append(real_val)
        y_pred.append(est_val)

    return np.array(y_true), np.array(y_pred)


##############################
# EXEMPLE D'UTILISATION
##############################
if __name__ == "__main__":
    np.random.seed(42)

    # 1) Génération de données
    t = np.linspace(0, 4*np.pi, 100)
    X = np.sin(t)
    Y = np.sin(t + np.pi/3) + 0.1*np.random.randn(len(t))  # "target" potentielle
    Z = np.cos(t) + 0.05*np.random.randn(len(t))

    # 2) Premier cas : Attracteur à partir de X seul
    #    => disons qu'on veut X(t), X(t-1), X(t-2)
    #    => on va tester si X reconstruit Y
    data_dict = {'x': X, 'y': Y, 'z': Z}
    
    lags_for_x_only = {
        'x': [0, 1, 2],  # 3 composantes => embedding dimension=3
        'y': [],         # pas de lags pour y => on ne l'utilise pas dans l'attracteur
        'z': []          # pas de lags pour z
    }

    emb_x, valid_t_x = build_multivariate_embedding(data_dict, lags_for_x_only)
    # On va visualiser ce nuage 3D (embedding) => correspond à (X(t), X(t-1), X(t-2))
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')
    ax1.scatter(emb_x[:, 0], emb_x[:, 1], emb_x[:, 2])
    ax1.set_title("Attracteur 3D reconstruit à partir de X (lags=0,1,2)")
    plt.show()

    # CCM : X -> Y
    y_true, y_est = ccm_reconstruct_multivariate(embedding=emb_x,
                                                 valid_time=valid_t_x,
                                                 effect_series=Y,
                                                 k=4,
                                                 use_weighted_mean=True)
    rho_x_y = correlation_coefficient(y_true, y_est)
    print(f"[X-only attracteur] Corrélation CCM(X->Y) = {rho_x_y:.3f}")

    # 3) Second cas : Attracteur multivarié à partir de [X, Z]
    #    => X(t), X(t-1), Z(t) => dimension=3
    #    => on teste la reconstruction de Y
    lags_for_xz = {
        'x': [0, 1],
        'z': [0],
        'y': []  # on n'utilise pas y dans l'attracteur
    }

    emb_xz, valid_t_xz = build_multivariate_embedding(data_dict, lags_for_xz)

    # Visualisation 3D => (X(t), X(t-1), Z(t))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')
    ax2.scatter(emb_xz[:, 0], emb_xz[:, 1], emb_xz[:, 2])
    ax2.set_title("Attracteur 3D reconstruit à partir de [X, Z]")
    plt.show()

    # CCM : [X, Z] -> Y
    y_true_xz, y_est_xz = ccm_reconstruct_multivariate(embedding=emb_xz,
                                                       valid_time=valid_t_xz,
                                                       effect_series=Y,
                                                       k=4,
                                                       use_weighted_mean=True)
    rho_xz_y = correlation_coefficient(y_true_xz, y_est_xz)
    print(f"[X+Z attracteur] Corrélation CCM(X,Z->Y) = {rho_xz_y:.3f}")
