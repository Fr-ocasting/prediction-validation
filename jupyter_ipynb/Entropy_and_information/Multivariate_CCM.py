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

def get_embedding_and_reconstruct(data_dict,target_variable,lags,k=4,use_weighted_mean=True):
    embedding, valid_time = build_multivariate_embedding(data_dict, lags)
    y_true, y_est = ccm_reconstruct_multivariate(embedding=embedding,
                                                    valid_time=valid_time,
                                                    effect_series=data_dict[target_variable],
                                                    k=k,
                                                    use_weighted_mean=use_weighted_mean)
    rho_x_y = correlation_coefficient(y_true, y_est)
    print(f"[X-only attracteur] Corrélation CCM(X->Y) = {rho_x_y:.3f}")
    return y_est,embedding

def plot_reconstruction(embedding,lags={}):
    fig1 = plt.figure()
    if embedding.shape[1]>=3:
        ax = fig1.add_subplot(projection='3d')
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2])
        if lags == {}:
            title = "Original Space (X(t),Y(t),Z(t))"
        else:
            title = f"Attracteur 3D reconstruit à partir de {', '.join([k for k,v in lags.items() if len(v)>0])} (lags={', '.join([f'{k}: {v}' for k,v in lags.items() if len(v)>0])})"
        ax.set_title(title)
        ax.view_init(elev=20, azim=10, roll=0)
    else:
        ax = fig1.add_subplot()
        ax.scatter(embedding[:, 0], embedding[:, 1])
        if lags == {}:
            title = "Original Space (X(t),Y(t))"
        else:
            title = f"Attracteur 2D reconstruit à partir de {', '.join([k for k,v in lags.items() if len(v)>0])} (lags={', '.join([f'{k}: {v}' for k,v in lags.items() if len(v)>0])})"
        ax.set_title(title)
    plt.show()



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

    data_dict = {'x': X, 'y': Y, 'z': Z}
    plot_reconstruction(np.array([x,y,z]).transpose())

    # 2) Premier cas : Attracteur à partir de X seul
    #    => disons qu'on veut X(t), X(t-1), X(t-2)
    #    => on va tester si X reconstruit Y
    lags = {'x': [0, 1, 2]}
    y_est,embedding = get_embedding_and_reconstruct(data_dict,target_variable='y',lags=lags,k=4,use_weighted_mean=True)
    plot_reconstruction(embedding,lags)


    # 3) Second cas : Attracteur multivarié à partir de [X, Z]
    #    => X(t), X(t-1), Z(t) => dimension=3
    #    => on teste la reconstruction de Y
    lags = {'x': [0, 1], 'z': [0]}
    y_est,embedding = get_embedding_and_reconstruct(data_dict,target_variable='y',lags=lags,k=4,use_weighted_mean=True)
    plot_reconstruction(embedding,lags)
