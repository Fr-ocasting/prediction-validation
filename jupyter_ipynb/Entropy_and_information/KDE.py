import numpy as np
from sklearn.neighbors import NearestNeighbors, KernelDensity


##############################################
# KDEEstimator
##############################################

class KDEEstimator:
    """
    Estimation par noyau (Kernel Density Estimation) pour l'entropie,
    l'information mutuelle, et la Transfer Entropy.
    """

    def __init__(self, bandwidth: float = 0.2):
        """
        :param bandwidth: Largeur de bande pour la densité gaussienne.
        """
        self.bandwidth = bandwidth

    def _entropy_kde(self, data: np.ndarray) -> float:
        """
        H(X) ~= - 1/N * sum_{i=1}^N log p(x_i), 
        p(.) estimé par KernelDensity (gaussien).
        
        => Estimation connue pour avoir un certain biais,
           mais on la laisse telle quelle par simplicité.
        """
        n, dim = data.shape
        if n < 2:
            return 0.0

        kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        kde.fit(data)
        logp = kde.score_samples(data)  # log p(x_i)
        return - np.mean(logp)

    def estimate_mi(self, data_x: np.ndarray, data_y: np.ndarray) -> float:
        """
        I(X;Y) = H(X) + H(Y) - H(X,Y).
        """
        xy = np.hstack([data_x, data_y])
        Hx = self._entropy_kde(data_x)
        Hy = self._entropy_kde(data_y)
        Hxy = self._entropy_kde(xy)
        return Hx + Hy - Hxy

    def estimate_conditional_mi(self, data_x: np.ndarray,
                                data_y: np.ndarray,
                                data_z: np.ndarray) -> float:
        """
        I(X; Y | Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z).
        """
        xz = np.hstack([data_x, data_z])
        yz = np.hstack([data_y, data_z])
        xyz = np.hstack([data_x, data_y, data_z])

        Hxz = self._entropy_kde(xz)
        Hyz = self._entropy_kde(yz)
        Hxyz = self._entropy_kde(xyz)
        Hz = self._entropy_kde(data_z)

        return Hxz + Hyz - Hxyz - Hz