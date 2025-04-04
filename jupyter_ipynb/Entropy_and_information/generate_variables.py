import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple

#####################
# 1) Système type Lorenz
#####################
def lorenz_system(state, sigma=10.0, rho=28.0, beta=8/3):
    """
    Équation de Lorenz :
    dx/dt = sigma*(y - x)
    dy/dt = x*(rho - z) - y
    dz/dt = x*y - beta*z
    """
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def simulate_lorenz(initial_state: np.ndarray,
                    dt: float = 0.01,
                    steps: int = 10000) -> np.ndarray:
    """
    Intégration d'Euler simple pour le système de Lorenz.
    Retourne un tableau (steps, 3) avec la trajectoire [x(t), y(t), z(t)].
    """
    traj = np.zeros((steps, 3))
    traj[0] = initial_state
    for i in range(1, steps):
        deriv = lorenz_system(traj[i-1])
        traj[i] = traj[i-1] + dt * deriv
    return traj

#####################
# 2) Génération des données "Moutons, Loups, Herbe"
#####################
np.random.seed(42)
initial_state = np.array([5.0, 5.0, 5.0])  # point de départ
data_lorenz = simulate_lorenz(initial_state, dt=0.01, steps=6000)
# Tronquons un peu le début pour éviter les transitoires
data_lorenz = data_lorenz[1000:]  # shape (5000, 3)

# On renomme : Moutons = X, Loups = Y, Herbe = Z
Moutons = data_lorenz[:, 0]
Loups = data_lorenz[:, 1]
Herbe = data_lorenz[:, 2]

#####################
# 3) Affichage 3D : "Papillon"
#####################
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(Moutons, Loups, Herbe, lw=0.5)
ax.set_xlabel("Moutons")
ax.set_ylabel("Loups")
ax.set_zlabel("Herbe")
ax.set_title("Trajectoire pseudo-chaotique (Lorenz) rebaptisée Moutons–Loups–Herbe")
plt.show()

#####################
# 4) Petit test CCM
#####################

# (a) On échantillonne la fin de la trajectoire (par ex. 1000 points) pour avoir la série
M = 1000
Moutons_sample = Moutons[-M:]
Loups_sample = Loups[-M:]
Herbe_sample = Herbe[-M:]

# (b) On applique CCM => vous pouvez réutiliser votre fonction ccm_reconstruct_multivariate
#     ou ccm_reconstruct mono-série. Ex. : Loups -> Moutons
#     On donne juste un squelette illustratif (simplifié) :

from sklearn.neighbors import NearestNeighbors

def ccm_reconstruct(cause_series: np.ndarray,
                    effect_series: np.ndarray,
                    E: int = 3,
                    tau: int = 1,
                    k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    CCM simple, monovarié. Reconstruit effect_series à partir de l'attracteur de cause_series.
    """
    N = len(cause_series)
    start_index = (E - 1) * tau
    # Attracteur
    M_points = []
    valid_t = []
    for t in range(start_index, N):
        coords = [cause_series[t - i*tau] for i in range(E)]
        M_points.append(coords)
        valid_t.append(t)
    M_points = np.array(M_points)
    valid_t = np.array(valid_t)

    # k-NN
    nbrs = NearestNeighbors(n_neighbors=k).fit(M_points)
    y_pred = []
    y_true = []
    for i, mp in enumerate(M_points):
        dist, idxs = nbrs.kneighbors(mp.reshape(1, -1))
        # Weighted average
        d = dist[0]
        d = np.where(d<1e-12, 1e-12, d)
        w = 1/d
        w /= w.sum()
        neighbors_t = valid_t[idxs[0]]
        val_neighbors = effect_series[neighbors_t]
        val_est = np.sum(val_neighbors * w)
        y_pred.append(val_est)
        # Valeur réelle
        t_i = valid_t[i]
        y_true.append(effect_series[t_i])
    return np.array(y_true), np.array(y_pred)

def corr(x, y):
    return np.corrcoef(x, y)[0, 1]

# Test : CCM(Loups->Moutons)
E_dim = 3
tau_lag = 2
k_nn = 6
y_true, y_est = ccm_reconstruct(Loups_sample, Moutons_sample,
                                E=E_dim, tau=tau_lag, k=k_nn)
r_loups_moutons = corr(y_true, y_est)
print(f"Corrélation CCM(Loups->Moutons) = {r_loups_moutons:.3f}")

# Test : CCM(Moutons->Loups)
y_true2, y_est2 = ccm_reconstruct(Moutons_sample, Loups_sample,
                                  E=E_dim, tau=tau_lag, k=k_nn)
r_moutons_loups = corr(y_true2, y_est2)
print(f"Corrélation CCM(Moutons->Loups) = {r_moutons_loups:.3f}")

# Idem si vous voulez tester Herbe -> Moutons, etc.

plt.figure()
plt.plot(y_true[:200], label="Moutons (vrai)", alpha=0.7)
plt.plot(y_est[:200], label="Moutons (estimé via CCM Loups->Moutons)", alpha=0.7)
plt.title("Extrait reconstruction CCM(Loups->Moutons)")
plt.legend()
plt.show()
