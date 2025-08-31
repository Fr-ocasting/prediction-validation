import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from typing import Tuple


import sys
import os

current_path = notebook_dir = os.getcwd()
working_dir = os.path.abspath(os.path.join(current_path, '..','..'))
if working_dir not in sys.path:
    sys.path.insert(0, working_dir)
    
from pipeline.jupyter_ipynb.Entropy_and_information.utils import correlation_coefficient

def ccm_reconstruct(cause_series: np.ndarray,
                    effect_series: np.ndarray,
                    E: int = 3,
                    tau: int = 1,
                    k: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convergent Cross Mapping (version simplifiée) :
    
    1) Construit l'attracteur M(t) pour la série 'cause_series' en dimension E, retard tau.
    2) Pour chaque t, on cherche les k plus proches voisins de M(t).
       On récupère effect_series[voisins], et on calcule la moyenne pondérée (ou simple) comme estimation de Y(t).
    3) On compare la valeur estimée à la valeur réelle effect_series[t].
    
    :param cause_series: Série (X) supposée "cause".
    :param effect_series: Série (Y) sur laquelle on veut vérifier l'influence de X.
    :param E: Dimension d'embedding (nombre de composantes dans chaque vecteur M(t)).
    :param tau: Pas de retard.
    :param k: Nombre de voisins pour la reconstruction.
    :return: (y_true_array, y_pred_array) pour ensuite calculer la corrélation.
    """

    N = len(cause_series)
    # Pour construire l'attracteur, on a besoin que i - (E-1)*tau >= 0
    # => i >= (E-1)*tau
    start_index = (E - 1) * tau
    M = []
    valid_times = []  # pour connaître l'instant t associé à chaque vecteur

    # Construction du manifold M
    for t in range(start_index, N):
        # vecteur = [ X(t), X(t - tau), ..., X(t - (E-1)*tau) ]
        coords = []
        for e in range(E):
            coords.append(cause_series[t - e*tau])
        coords = np.array(coords)
        M.append(coords)
        valid_times.append(t)

    M = np.vstack(M)  # shape (M_points, E)
    # M_points = N - start_index
    M_points = M.shape[0]

    # Fit NN
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(M)

    # Estimation de effect_series pour chacun des points M(t)
    y_pred = []
    y_true = []
    for i in range(M_points):
        # On s'intéresse au vrai temps t = valid_times[i]
        t = valid_times[i]
        # Chercher les k plus proches voisins de M[i]
        distances, indices = nbrs.kneighbors(M[i].reshape(1, -1))
        # indices[0] = liste des k plus proches voisins
        # On peut faire une moyenne simple des effect_series correspondants, ou une moyenne pondérée 1/d.
        
        # Récupérer Y[j] pour j dans indices[0]
        ys = []
        w = []
        for idx_nb in indices[0]:
            t_nb = valid_times[idx_nb]  # le temps correspondant
            ys.append(effect_series[t_nb])
            dist = np.linalg.norm(M[i] - M[idx_nb])
            # Eviter division par 0
            if dist < 1e-12:
                dist = 1e-12
            w.append(1.0/dist)
        
        ys = np.array(ys)
        w = np.array(w)
        w = w / w.sum()  # normalisation
        y_hat = np.sum(ys * w)
        
        y_pred.append(y_hat)
        # Valeur réelle
        y_true.append(effect_series[t])

    return np.array(y_true), np.array(y_pred)



if __name__ == "__main__":
    # ---------------------------
    # 1) Données d'exemple
    # ---------------------------
    np.random.seed(42)
    t = np.linspace(0, 4*np.pi, 100)
    X = np.sin(t)                    # "cause" potentielle
    Y = np.sin(t + np.pi/3)          # "effet" potentiellement causé par X
    Z = np.cos(t) + 0.2*np.random.randn(len(t))  # Autre variable

    # ---------------------------
    # 2) Graphique 3D (X,Y,Z) original
    # ---------------------------
    # Plot de l'espace "réel" (x(t), y(t), z(t)) 
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')
    ax1.scatter(X, Y, Z)
    ax1.set_title("Espace original : (X(t), Y(t), Z(t))")
    plt.show()

    # ---------------------------
    # 3) Reconstruction d'attracteur en 3D à partir de X(t)
    # ---------------------------
    # Prenons E=3, tau=2 pour l'illustration
    E = 3
    tau = 2
    N = len(X)
    start_index = (E - 1)*tau
    embedding = []
    for i in range(start_index, N):
        coords = [X[i - e*tau] for e in range(E)]
        embedding.append(coords)
    embedding = np.array(embedding)  # shape (N - start_index, 3)

    # Plot 3D de l'attracteur reconstruit
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')
    ax2.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2])
    ax2.set_title("Attracteur reconstruit à partir de X(t), E=3, tau=2")
    plt.show()

    # ---------------------------
    # 4) Convergent Cross Mapping: X -> Y
    # ---------------------------
    # Paramètres de CCM
    ccm_dim = 3
    ccm_tau = 1
    ccm_k = 4

    # On obtient y_true, y_est
    y_true, y_est = ccm_reconstruct(
        cause_series=X,
        effect_series=Y,
        E=ccm_dim,
        tau=ccm_tau,
        k=ccm_k
    )

    # Corrélation
    rho = correlation_coefficient(y_true, y_est)
    print(f"Corrélation CCM (X -> Y) = {rho:.3f}")

    # Interprétation du signe
    if rho > 0.1:  # Seuil arbitraire
        print("CCM suggère que X a une influence positive sur Y (corrélation > 0).")
    else:
        print("Aucune influence notable détectée de X vers Y (corrélation faible ou nulle).")

    # ---------------------------
    # 5) Convergent Cross Mapping: Y -> X (test inverse)
    # ---------------------------
    x_true, x_est = ccm_reconstruct(
        cause_series=Y,
        effect_series=X,
        E=ccm_dim,
        tau=ccm_tau,
        k=ccm_k
    )
    rho_yx = correlation_coefficient(x_true, x_est)
    print(f"Corrélation CCM (Y -> X) = {rho_yx:.3f}")
    if rho_yx > 0.1:
        print("CCM suggère que Y a une influence positive sur X.")
    else:
        print("Aucune influence notable détectée de Y vers X.")
