import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma

##############################################
# KSGEstimator
##############################################

class KSGEstimator:
    """
    Estimation non-paramétrique (KSG) de l'entropie / information mutuelle / Transfer Entropy
    basée sur le principe k-Nearest Neighbors.
    
    - Si `simplified=True`, on fait une version "naïve" utilisant simplement
      H(X) ~ digamma(N) - digamma(k) + dim * mean(log(k-th_distance)).
    - Si `simplified=False`, on applique une formule KSG1 plus standard :
      I(X;Y) = digamma(k) - (1/N)*sum_{i=1}^N [digamma(n_x(i)+1) + digamma(n_y(i)+1)] + ...
      (voir Kraskov et al. 2004).
    """
    def __init__(self, k: int = 5, simplified: bool = True):
        """
        :param k: nombre de plus proches voisins
        :param simplified: si True, version simplifiée, sinon version KSG1 plus complète
        """
        self.k = k
        self.simplified = simplified

    ###################################################
    # Outils internes : KSG complet (KSG1)
    ###################################################
    @staticmethod
    def _count_neighbors_in_radius(data: np.ndarray, radius: np.ndarray) -> np.ndarray:
        """
        Pour chaque point i, compte le nombre de points j != i dont la distance
        à i dans 'data' est < radius[i].
        
        data: shape (N, d)
        radius: shape (N,) rayon pour chaque point
        return: array (N,) nombre de voisins (excluant i) dans ce rayon
        """
        # On va construire un NearestNeighbors global, puis pour chaque point i
        # on compte combien se trouvent à distance < radius[i].
        N = data.shape[0]
        nbrs = NearestNeighbors(n_neighbors=N, algorithm='auto').fit(data)
        distances, indices = nbrs.kneighbors(data)
        # distances[i,:] = distances from i to all sorted neighbors
        # on veut n_x(i) = le nombre j != i tel que distance[i,j] < radius[i]
        
        counts = np.zeros(N, dtype=int)
        for i in range(N):
            # on compte combien de distances < radius[i]
            # distances[i,0] = 0 (c'est le point i lui-même)
            # On peut faire:
            c = np.sum(distances[i, :] < radius[i])
            # On retire 1 pour exclure le point lui-même
            counts[i] = c - 1
        return counts

    def _entropy_ksg_full(self, data: np.ndarray) -> float:
        """
        H(X) selon la formule KSG "complète" (version "KSG1" appliquée à la single-entropy).
        Dans Kraskov (2004), la formule est plus orientée sur I(X;Y).
        Ici, on fait un usage partiel pour H(X) (-> KSG n'est pas toujours 1-to-1 pour H(X)).
        
        NOTE : KSG est surtout défini pour I(X;Y), on l'étend ici à H(X) en
        comptant le rayon sur X, puis n_x(i) etc. 
        Cela reste un "workaround" car KSG original se focalise plus sur la MI.
        
        On va tout de même faire la version "simplifiée" pour H(X) ou la "leave-one-out".
        Cf. Kozachenko & Leonenko (1987) ou "KSG" unidimensionnel, on applique :
        
        H(X) ~= digamma(N) - digamma(k) + d * 1/N * sum( log( c_d * distance_k(i)^2 ) ) + ...
        (On laisse la const c_d de côté, souvent c_d = pi pour d=2, etc.)
        
        Pour rester cohérent, on va faire la version "Kozachenko & Leonenko" ou "Kraskov single-entropy" en 1 passe.
        """
        # => Pour la "vraie" KSG "full" sur l'entropie, il y a de multiples formules possibles,
        #    toutes plus ou moins approximatives. On va en prendre une version "usuelle" :
        
        N, dim = data.shape
        if N <= self.k:
            return 0.0
        
        # Fit NN
        nbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='auto').fit(data)
        distances, _ = nbrs.kneighbors(data)
        # distances[i, 0] = 0, distances[i, self.k] = dist to k-th neighbor
        kth_dist = distances[:, self.k]  # shape (N,)

        # Kozachenko-Leonenko approximation:
        # H(X) ~ digamma(N) - digamma(k) + d * (1/N) * sum( log(2 * kth_dist_i) ) + const
        #    On ignore la constante du volume de la boule unitaire en dim d, car
        #    pour du calcul de MI, elle s'annule souvent. Mais si on veut l'entropie absolue
        #    on devrait l'ajouter. On la laisse de côté pour la cohérence.
        
        eps = 1e-15
        avg_log_dist = np.mean(np.log(kth_dist + eps))
        hx = digamma(N) - digamma(self.k) + dim * avg_log_dist
        return float(hx)

    def _mi_ksg_full(self, data_x: np.ndarray, data_y: np.ndarray) -> float:
        """
        I(X;Y) selon la formule KSG1 (Kraskov 2004).
        
        I(X;Y) = digamma(k) - 1/N sum_i [ digamma(n_x(i)+1) + digamma(n_y(i)+1 ) ] + digamma(N)
        
        Où n_x(i) = #points dont la distance en X est < e_i,
            e_i = 0.5 * distance dans l'espace (X,Y) au k-ième plus proche voisin de i.
        
        NB: Il existe 2 versions (KSG1, KSG2). On code la KSG1 ici.
        """
        N = data_x.shape[0]
        if data_y.shape[0] < N:
            N = data_y.shape[0]
        data_x = data_x[:N]
        data_y = data_y[:N]
        
        xy = np.hstack([data_x, data_y])  # shape (N, dx+dy)
        N, dxy = xy.shape
        
        if N <= self.k:
            return 0.0

        # Fit NN in XY
        nbrs_xy = NearestNeighbors(n_neighbors=self.k+1, algorithm='auto').fit(xy)
        dist_xy, _ = nbrs_xy.kneighbors(xy)
        kth_xy = dist_xy[:, self.k]  # distance au k-ième plus proche voisin en XY

        # e_i = (kth_xy[i] / 2) ou, plus souvent, on prend la distance strictement < kth_xy[i] ?
        # On fera la convention e_i = kth_xy[i] + un epsilon => 
        # Dans la littérature, e_i est souvent 1/2 * kth_xy[i], ici on fera e_i = kth_xy[i] sans diviser par 2,
        # car on se focalise sur le compte n_x(i), n_y(i) < e_i. De multiples variantes existent.
        
        e_xy = kth_xy
        
        # Now on compte n_x(i) = nb de points (j != i) dont la distance en X est < e_xy[i]
        # idem pour n_y(i). 
        # On calcule d'abord la distance en X:
        dx_nbrs = NearestNeighbors(n_neighbors=N, algorithm='auto').fit(data_x)
        dist_x, _ = dx_nbrs.kneighbors(data_x)

        dy_nbrs = NearestNeighbors(n_neighbors=N, algorithm='auto').fit(data_y)
        dist_y, _ = dy_nbrs.kneighbors(data_y)
        
        # Pour i, on cherche la distance < e_xy[i], on compte => n_x(i).
        n_x_vals = np.zeros(N, dtype=int)
        n_y_vals = np.zeros(N, dtype=int)

        for i in range(N):
            # dist_x[i, :] = distances de i à tous
            n_x = np.sum(dist_x[i, :] < e_xy[i]) - 1  # exclure i
            n_y = np.sum(dist_y[i, :] < e_xy[i]) - 1
            n_x_vals[i] = n_x
            n_y_vals[i] = n_y

        # I(X;Y) = digamma(k) + digamma(N) - (1/N)*sum_i [ digamma(n_x(i)+1) + digamma(n_y(i)+1 ) ]
        # doc: Kraskov 2004 eq. (8)
        kf = float(self.k)
        return digamma(kf) + digamma(N) - np.mean(digamma(n_x_vals+1) + digamma(n_y_vals+1))

    ###################################################
    # Méthodes "simplifiées"
    ###################################################
    def _entropy_ksg_simplified(self, data: np.ndarray) -> float:
        """
        H(X) ~ digamma(N) - digamma(k) + dim * mean( log(kth_neighbor_distance) )
        => ignoring constant volume terms
        """
        N, dim = data.shape
        if N <= self.k:
            return 0.0

        nbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='auto').fit(data)
        distances, _ = nbrs.kneighbors(data)
        kth_dist = distances[:, self.k]
        eps = 1e-15
        avg_log_dist = np.mean(np.log(kth_dist + eps))
        hx = digamma(N) - digamma(self.k) + dim * avg_log_dist
        return float(hx)

    def _mi_ksg_simplified(self, data_x: np.ndarray, data_y: np.ndarray) -> float:
        """
        I(X;Y) = H(X) + H(Y) - H(X,Y), 
        avec H(...) calculé via la version _entropy_ksg_simplified.
        """
        xy = np.hstack([data_x, data_y])
        Hx = self._entropy_ksg_simplified(data_x)
        Hy = self._entropy_ksg_simplified(data_y)
        Hxy = self._entropy_ksg_simplified(xy)
        return Hx + Hy - Hxy

    ###################################################
    # API publique
    ###################################################
    def _entropy(self, data: np.ndarray) -> float:
        if self.simplified:
            return self._entropy_ksg_simplified(data)
        else:
            return self._entropy_ksg_full(data)

    def estimate_mi(self, data_x: np.ndarray, data_y: np.ndarray) -> float:
        """
        I(X;Y) = H(X) + H(Y) - H(X,Y)
        Soit en version KSG simplifiée, soit KSG1 "complète".
        """
        if self.simplified:
            return self._mi_ksg_simplified(data_x, data_y)
        else:
            return self._mi_ksg_full(data_x, data_y)

    def estimate_conditional_mi(self, data_x: np.ndarray,
                                data_y: np.ndarray,
                                data_z: np.ndarray) -> float:
        """
        I(X; Y | Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z).
        """
        xz = np.hstack([data_x, data_z])
        yz = np.hstack([data_y, data_z])
        xyz = np.hstack([data_x, data_y, data_z])

        Hxz = self._entropy(xz)
        Hyz = self._entropy(yz)
        Hxyz = self._entropy(xyz)
        Hz = self._entropy(data_z)

        return Hxz + Hyz - Hxyz - Hz


