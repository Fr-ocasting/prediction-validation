import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect
import os, sys
sys.path.append(os.path.abspath('../..'))
from pipeline.jupyter_ipynb.Entropy_and_information.granger import GrangerCausalityAnalysis

def generate_linear_causal_series(n=200, lags=[2, 4], coeffs=[0.6, 0.3], noise_level=0.2, seed=None):
    """
    Generate time series with a clear linear causal relationship: X -> Y
    
    Parameters:
    -----------
    n : int
        Number of time points
    lags : list
        List of lags at which X influences Y
    coeffs : list
        Coefficients for each lag (strength of influence)
    noise_level : float
        Standard deviation of noise
    seed : int, optional
        Random seed
        
    Returns:
    --------
    t : array
        Time points
    x : array
        Cause variable X
    y : array
        Effect variable Y (influenced by X)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Time points
    t = np.arange(n)
    
    # Generate X as a stationary AR(1) process
    x = np.zeros(n)
    x[0] = np.random.randn()
    for i in range(1, n):
        x[i] = 0.5 * x[i-1] + np.random.randn() * noise_level
    
    # Generate Y influenced by lagged values of X
    y = np.zeros(n)
    y[0] = np.random.randn() * noise_level
    
    max_lag = max(lags)
    for i in range(1, n):
        # Autoregressive component for Y
        y[i] = 0.2 * y[i-1] + np.random.randn() * noise_level
        
        # Add influence from X at specified lags
        for lag, coeff in zip(lags, coeffs):
            if i >= lag:
                y[i] += coeff * x[i-lag]
    
    # Expected result: X Granger-causes Y at the specified lags
    print(f"Expected Granger causality:\nX -> Y at lags {lags}\n\nNo Granger causality:\nY -> X")

    return t, x, y

def generate_bidirectional_causality(n=200, xy_lags=[2], xy_coeffs=[0.4], 
                                    yx_lags=[3], yx_coeffs=[0.2], 
                                    noise_level=0.2, seed=None):
    """
    Generate time series with bidirectional causality: X <-> Y with different strengths
    
    Parameters:
    -----------
    n : int
        Number of time points
    xy_lags, yx_lags : list
        Lags for X->Y and Y->X influences
    xy_coeffs, yx_coeffs : list
        Coefficients for each lag (typically xy_coeffs > yx_coeffs for X->Y stronger than Y->X)
    noise_level : float
        Standard deviation of noise
    seed : int, optional
        Random seed
        
    Returns:
    --------
    t : array
        Time points
    x : array
        X variable
    y : array
        Y variable
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(n)
    x = np.zeros(n)
    y = np.zeros(n)
    
    # Initialize with random values
    x[0] = np.random.randn() * noise_level
    y[0] = np.random.randn() * noise_level
    
    # Maximum lags
    max_xy_lag = max(xy_lags) if xy_lags else 0
    max_yx_lag = max(yx_lags) if yx_lags else 0
    max_lag = max(max_xy_lag, max_yx_lag)
    
    # Fill in first values where lags don't apply yet
    for i in range(1, max_lag+1):
        x[i] = 0.3 * x[i-1] + np.random.randn() * noise_level
        y[i] = 0.3 * y[i-1] + np.random.randn() * noise_level
    
    # Generate series with bidirectional influences
    for i in range(max_lag+1, n):
        # X with influence from Y
        x[i] = 0.3 * x[i-1] + np.random.randn() * noise_level
        for lag, coeff in zip(yx_lags, yx_coeffs):
            x[i] += coeff * y[i-lag]
        
        # Y with influence from X
        y[i] = 0.3 * y[i-1] + np.random.randn() * noise_level
        for lag, coeff in zip(xy_lags, xy_coeffs):
            y[i] += coeff * x[i-lag]
    
    # Expected results
    print(f"Expected Granger causality:\nX -> Y at lags {xy_lags}\n Y -> X at lags {yx_lags}")
    print(f"Stronger causality: {'X -> Y' if max(xy_coeffs) > max(yx_coeffs) else 'Y -> X'}")
    
    return t, x, y

def generate_common_cause(n=200, xz_lags=[2], xz_coeffs=[0.6], 
                          yz_lags=[3], yz_coeffs=[0.6],
                          noise_level=0.2, seed=None):
    """
    Generate time series where Z is a common cause for both X and Y: Z -> X and Z -> Y
    No direct causal link between X and Y.
    
    Parameters:
    -----------
    n : int
        Number of time points
    xz_lags, yz_lags : list
        Lags for Z->X and Z->Y influences
    xz_coeffs, yz_coeffs : list
        Coefficients for each lag
    noise_level : float
        Standard deviation of noise
    seed : int, optional
        Random seed
        
    Returns:
    --------
    t : array
        Time points
    x : array
        X variable (affected by Z)
    y : array
        Y variable (affected by Z)
    z : array
        Z variable (common cause)
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(n)
    
    # Generate Z as a stationary AR(1) process
    z = np.zeros(n)
    z[0] = np.random.randn()
    for i in range(1, n):
        z[i] = 0.5 * z[i-1] + np.random.randn() * noise_level
    
    # Initialize X and Y
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = np.random.randn() * noise_level
    y[0] = np.random.randn() * noise_level
    
    # Maximum lags
    max_xz_lag = max(xz_lags) if xz_lags else 0
    max_yz_lag = max(yz_lags) if yz_lags else 0
    max_lag = max(max_xz_lag, max_yz_lag)
    
    # Fill in first values
    for i in range(1, max_lag+1):
        x[i] = 0.3 * x[i-1] + np.random.randn() * noise_level
        y[i] = 0.3 * y[i-1] + np.random.randn() * noise_level
    
    # Generate X and Y influenced by Z
    for i in range(max_lag+1, n):
        # X influenced by Z
        x[i] = 0.3 * x[i-1] + np.random.randn() * noise_level
        for lag, coeff in zip(xz_lags, xz_coeffs):
            x[i] += coeff * z[i-lag]
        
        # Y influenced by Z
        y[i] = 0.3 * y[i-1] + np.random.randn() * noise_level
        for lag, coeff in zip(yz_lags, yz_coeffs):
            y[i] += coeff * z[i-lag]
    
    # Expected results
    print("Expected Granger causality:")
    print(f"Z -> X at lags {xz_lags}")
    print(f"Z -> Y at lags {yz_lags}")
    print("\nNo direct Granger causality:\nX and Y")
    print("However, false positive X -> Y or Y -> X may appear due to common cause")
    
    return t, x, y, z

def generate_nonlinear_causality(n=200, lag=2, noise_level=0.2, seed=None):
    """
    Generate time series with a non-linear causal relationship: X -> Y
    
    Parameters:
    -----------
    n : int
        Number of time points
    lag : int
        Lag at which X influences Y
    noise_level : float
        Standard deviation of noise
    seed : int, optional
        Random seed
        
    Returns:
    --------
    t : array
        Time points
    x : array
        Cause variable X
    y : array
        Effect variable Y (non-linearly influenced by X)
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(n)
    
    # Generate X as a stationary AR(1) process
    x = np.zeros(n)
    x[0] = np.random.randn()
    for i in range(1, n):
        x[i] = 0.5 * x[i-1] + np.random.randn() * noise_level
    
    # Generate Y influenced by non-linear function of X
    y = np.zeros(n)
    y[0] = np.random.randn() * noise_level
    
    for i in range(1, n):
        # Autoregressive component
        y[i] = 0.3 * y[i-1] + np.random.randn() * noise_level
        
        # Add non-linear influence from X
        if i >= lag:
            y[i] += 0.5 * np.sin(x[i-lag]) + 0.3 * x[i-lag]**2
    
    print(f"Expected non-linear Granger causality:\nX -> Y at lag {lag}")
    print("\nStandard linear Granger test might not fully capture this relationship")
    
    return t, x, y

def generate_seasonal_causality(n=300, period=24, lag=6, seasonal_strength=0.8, 
                              causal_strength=0.5, noise_level=0.2, seed=None):
    """
    Generate seasonal time series with causality: X -> Y
    X has strong seasonality, and influences Y with a lag
    
    Parameters:
    -----------
    n : int
        Number of time points
    period : int
        Seasonal period (e.g., 24 for daily data)
    lag : int
        Lag at which X influences Y
    seasonal_strength : float
        Strength of seasonal component
    causal_strength : float
        Strength of causal influence
    noise_level : float
        Standard deviation of noise
    seed : int, optional
        Random seed
        
    Returns:
    --------
    t : array
        Time points
    x : array
        Seasonal cause variable X
    y : array
        Effect variable Y
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(n)
    
    # Generate X with seasonality + trend + noise
    trend = 0.01 * t
    seasonality = seasonal_strength * np.sin(2 * np.pi * t / period)
    noise = np.random.normal(0, noise_level, n)
    
    x = trend + seasonality + noise
    
    # Generate Y influenced by lagged X + its own seasonality
    y = np.zeros(n)
    y_seasonality = 0.3 * np.sin(2 * np.pi * t / period + np.pi/3)  # Different phase
    
    for i in range(n):
        # Y has some seasonality too
        y[i] = y_seasonality[i] + np.random.normal(0, noise_level)
        
        # Add influence from X with lag
        if i >= lag:
            y[i] += causal_strength * x[i-lag]
    
    print(f"Expected Granger causality:\nX -> Y at lag {lag}")
    print("\nBoth X and Y have seasonality - needs differencing to be stationary")
    
    return t, x, y

def generate_intermittent_causality(n=400, active_periods=[(50, 100), (200, 250)], 
                                   lag=2, causal_strength=0.6, noise_level=0.2, seed=None):
    """
    Generate time series where X -> Y only during specific time periods
    
    Parameters:
    -----------
    n : int
        Number of time points
    active_periods : list of tuples
        List of (start, end) periods where causality is active
    lag : int
        Lag at which X influences Y
    causal_strength : float
        Strength of causal influence during active periods
    noise_level : float
        Standard deviation of noise
    seed : int, optional
        Random seed
        
    Returns:
    --------
    t : array
        Time points
    x : array
        Cause variable X
    y : array
        Effect variable Y (intermittently influenced by X)
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(n)
    
    # Generate X as a stationary AR(1) process
    x = np.zeros(n)
    x[0] = np.random.randn()
    for i in range(1, n):
        x[i] = 0.5 * x[i-1] + np.random.randn() * noise_level
    
    # Generate Y with intermittent influence from X
    y = np.zeros(n)
    y[0] = np.random.randn() * noise_level
    
    # Create mask for active periods
    is_active = np.zeros(n, dtype=bool)
    for start, end in active_periods:
        is_active[start:end] = True
    
    for i in range(1, n):
        # Autoregressive component
        y[i] = 0.3 * y[i-1] + np.random.randn() * noise_level
        
        # Add influence from X only during active periods
        if i >= lag and is_active[i]:
            y[i] += causal_strength * x[i-lag]
    
    print(f"Expected intermittent Granger causality: X -> Y at lag {lag}")
    print(f"Active during periods: {active_periods}")
    print("Standard Granger test on full series might show weaker causality")
    
    return t, x, y

# Helper function to demonstrate and test
def test_granger_with_generated_data(data_generator, params, max_lag=10):
    """
    Generate data, visualize it, and run Granger causality test
    """
    # Generate data
    sig = inspect.signature(data_generator)
    args_generator = {k: v for k, v in params.items() if k in sig.parameters}
    result = data_generator(**args_generator)
    t = result[0]
    
    # Create DataFrame
    columns = ['x', 'y'] if len(result) == 3 else ['x', 'y', 'z']
    df = pd.DataFrame(dict(zip(columns, result[1:])), index=t)
    
    # Plot
    plt.figure(figsize=(12, 6))
    df.plot()
    plt.title(f"Generated time series: {data_generator.__name__}")
    plt.show()
    
    # Run Granger causality analysis
    gc = GrangerCausalityAnalysis(df)
    criterion = params['criterion'] if 'criterion' in params else 'BIC'
    results = gc.full_analysis(max_lag=max_lag,criterion=criterion)
    
    return df, results


    
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

