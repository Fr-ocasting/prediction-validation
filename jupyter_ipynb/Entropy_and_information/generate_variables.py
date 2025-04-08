import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


    
# ----------------------------------------------------
# Lorenz
# ----------------------------------------------------
def lorenz_derivs(state, sigma=10.0, rho=28.0, beta=8/3):
    """
    Calcule dx/dt, dy/dt, dz/dt pour le système de Lorenz classique.
    state: (x, y, z).
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

def simulate_lorenz_noisy(initial_state, dt=0.01, steps=10000,
                          sigma=10.0, rho=28.0, beta=8/3,
                          noise_std=0.1):
    """
    Simule le système de Lorenz avec un terme de bruit additif à chaque pas.
    Schéma d'Euler simple.
    
    :param initial_state: np.array([x0, y0, z0])
    :param dt: pas de temps
    :param steps: nombre d'itérations
    :param sigma, rho, beta: paramètres du Lorenz
    :param noise_std: écart-type du bruit Gaussien à chaque pas
    :return: array shape (steps, 3) contenant la trajectoire
    """
    traj = np.zeros((steps, 3))
    traj[0] = initial_state
    
    for i in range(1, steps):
        x, y, z = traj[i-1]
        # dérivées déterministes
        dxdt, dydt, dzdt = lorenz_derivs([x, y, z], sigma, rho, beta)
        
        # Bruit Gaussien additif
        noise = np.random.normal(0, noise_std, size=3)
        
        # Schéma d'Euler: state_{n+1} = state_n + deriv * dt + noiseTerm
        # On ajoute le bruit comme s'il était un forcing (façon Euler-Maruyama simplifiée).
        traj[i] = traj[i-1] + dt * np.array([dxdt, dydt, dzdt]) + noise

    return traj


def load_variables_lorenz(steps = 6000,skip = 2000,sigma_lorenz = 10.0,rho_lorenz = 28.0,beta_lorenz = 8.0/3.0,
                   initial_state = np.array([5.0, 5.0, 5.0]),dt = 0.01,
                   noise_std = 0.2 , seed=42):
    np.random.seed(seed)
    trajectory = simulate_lorenz_noisy(
        initial_state=initial_state,
        dt=dt,
        steps=steps,
        sigma=sigma_lorenz,
        rho=rho_lorenz,
        beta=beta_lorenz,
        noise_std=noise_std
    )

    # On coupe le début (transitoire)
    
    data_lorenz = trajectory[skip:]  # shape (10000, 3) environ

    # Séparation en x, y, z
    x_noisy = data_lorenz[:, 0]
    y_noisy = data_lorenz[:, 1]
    z_noisy = data_lorenz[:, 2]
    return x_noisy,y_noisy,z_noisy
    
def plot_lorenz(x_noisy,y_noisy,z_noisy):
    fig = plt.figure(figsize=(12, 5))

    # a) Courbes x(t), y(t), z(t)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x_noisy, label="x(t) [Moutons]", alpha=0.8)
    ax1.plot(y_noisy, label="y(t) [Loups]", alpha=0.8)
    ax1.plot(z_noisy, label="z(t) [Herbe]", alpha=0.8)
    ax1.set_title("Évolution temporelle (Lorenz bruité)")
    ax1.legend()

    # b) Tracé 3D du papillon "imparfait"

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(x_noisy, y_noisy, z_noisy, lw=0.8)
    ax2.set_title("Attracteur de Lorenz bruité (3D)")
    ax2.set_xlabel("Moutons")
    ax2.set_ylabel("loup")
    ax2.set_zlabel("Herbe")

    plt.tight_layout()
    plt.show()

# ----------------------------------------------------
# TS with specific event which are related 
# ----------------------------------------------------

def load_variables_with_lagged_peak(seed=123,max_lag = 6, same_amplitude_through_lag = False,same_amplitude_per_lag=False,random_amplitude=False):
    np.random.seed(seed)

    # Paramètres
    days = 30
    hours_per_day = 24
    N = days * hours_per_day  # 720

    # On crée deux séries
    X_lagged_peak = np.random.rand(N) * 20  # baseline random
    Y_random_peak = np.random.rand(N) * 10

    # On ajoute des pics dans Y et des pics correspondants dans X avec des décalages variés
    amplitudes = np.arange(20,100)
    np.random.shuffle(amplitudes)
    for d in range(days):
        # Index jour = d * 24 à d*24+23
        start = d * hours_per_day

        lag = np.random.randint(1, max_lag)  # Décalage de 1 à 5 heures
        
        # Pic dans Y (à un moment aléatoire de la journée)
        y_peak_hour = np.random.randint(6, 22)  # Pic entre 6h et 22h
        y_peak_idx = start + y_peak_hour
        x_peak_idx = start + y_peak_hour - lag  # X a un pic avant Y

        if same_amplitude_through_lag:
            Y_random_peak[y_peak_idx] += 30.0  # Pic dans Y
            if x_peak_idx >= start:
                X_lagged_peak[x_peak_idx] += 50.0  # Pic plus fort dans X
        if same_amplitude_per_lag:
            Y_random_peak[y_peak_idx] += amplitudes[lag]
            if x_peak_idx >= start:
                X_lagged_peak[x_peak_idx] += amplitudes[lag]*5/3  # Pic plus fort dans X
        if random_amplitude:
            Y_random_peak[y_peak_idx] += np.random.randint(20, 100)
            if x_peak_idx >= start:
                X_lagged_peak[x_peak_idx] += np.random.randint(20, 100)  # Pic plus fort dans X

        
        # S'assurer que l'index est valide (ne pas dépasser le jour précédent)

    return X_lagged_peak,Y_random_peak

def load_variables_subway_bar(seed=123):
    np.random.seed(seed)

    # Paramètres
    days = 30
    hours_per_day = 24
    N = days * hours_per_day  # 720

    # On crée deux séries
    X_metro = np.random.rand(N) * 20  # baseline random
    Y_alcool = np.random.rand(N) * 10

    # Chaque jour, on fait un pic dans X vers 18h, un pic dans Y vers 19h
    for d in range(days):
        # index jour = d * 24 à d*24+23
        start = d * hours_per_day
        # pic X à 18h
        x_peak_idx = start + 18
        X_metro[x_peak_idx] += 50.0  # gros pic
        # pic Y à 19h
        y_peak_idx = start + 19
        Y_alcool[y_peak_idx] += 30.0
    return X_metro,Y_alcool



def plot_subway_bar(X_metro,Y_alcool):
    # Visualisation
    plt.figure(figsize=(12,4))
    plt.plot(X_metro, label="Passagers Métro (X)")
    plt.plot(Y_alcool, label="Consommation Bar (Y)")
    plt.title("Séries journalières : pic X à 18h -> pic Y à 19h")
    plt.legend()
    plt.show()

# ----------------------------------------------------
# TS Sinusoidales 
# ----------------------------------------------------
def load_variables_sinusoidales(n=100,T=2*np.pi,lag=np.pi/3,noise=True,seed=42,cos=True):
    """
    Génère 3 séries temporelles sinusoïdales avec un décalage de phase et du bruit.
    :param n: nombre de points
    :param T: longueur de la série
    :param lag: décalage de phase
    :param noise: si True, ajoute du bruit
    :param seed: pour la reproductibilité
    :param cos: si True, la série z est cos(t), sinon sin(t+2*lag)
    :return: t, x, y, z
    """
    np.random.seed(seed)
    t = np.linspace(0, T, n)  # n points from 0 to T
    y = np.sin(t)       
    if noise:
        x = np.sin(t + lag) + 0.05 * np.random.randn(len(t)) # décalage en phase + bruit
        if cos:
            z = np.cos(t) + 0.05 * np.random.randn(len(t))
        else:
            z = np.sin(t+2*lag) + 0.05 * np.random.randn(len(t))  # autre série + bruit 
    else:
        x = np.sin(t + lag)
        if cos:
            z = np.cos(t)
        else:
            z = np.sin(t+2*lag)
    return t,x,y,z

def load_sinusoidales_sum(n=100,T=2*np.pi,lag=np.pi/3,noise=True,seed=42,alpha= [0.5,0.5,3]):
    """
    Génère 3 séries temporelles: x et z sont des sinusoïdes, 
    y est la somme de x, z et d'une troisième sinusoïde.
    :param n: nombre de points
    :param T: longueur de la série
    :param lag: décalage de phase
    :param noise: si True, ajoute du bruit
    :param seed: pour la reproductibilité
    :return: t, x, y, z
    """
    np.random.seed(seed)
    t = np.linspace(0, T, n)  # n points from 0 to T

    x = np.sin(t + lag)
    z = np.sin(2*t + 2*lag)
    
    third_sin = np.sin(3*t)
    
    # y est la somme de x, z et third_sin
    y = alpha[0]*np.sin(t) + alpha[1]*np.sin(2*t) + alpha[2]*third_sin
    
    # Ajout du bruit si demandé
    if noise:
        x += 0.05 * np.random.randn(len(t))
        z += 0.05 * np.random.randn(len(t))
        y += 0.05 * np.random.randn(len(t))
    return t,x,y,z
# ----------------------------------------------------
# Exemple d'utilisation
# ----------------------------------------------------
if __name__ == "__main__":
    
    x_noisy,y_noisy,z_noisy = load_variables(steps = 6000,skip = 2000,sigma_lorenz = 10.0,rho_lorenz = 28.0,beta_lorenz = 8.0/3.0,
                    initial_state = np.array([5.0, 5.0, 5.0]),dt = 0.01,
                    noise_std = 0.2 , seed=42)
    plot_lorenz(x_noisy,y_noisy,z_noisy)

