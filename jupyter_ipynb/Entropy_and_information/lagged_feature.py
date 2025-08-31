import numpy as np 
import pandas as pd
from typing import Union, List, Dict

import sys
import os

current_path = notebook_dir = os.getcwd()
working_dir = os.path.abspath(os.path.join(current_path, '..','..'))
if working_dir not in sys.path:
    sys.path.insert(0, working_dir)

from pipeline.jupyter_ipynb.Entropy_and_information.KDE import KDEEstimator
from pipeline.jupyter_ipynb.Entropy_and_information.KSG import KSGEstimator
#############################
# BUILD LAGGED FEATURES
#############################
def build_lagged_features(series: np.ndarray, lags: List[int]) -> np.ndarray:
    """
    Construit un tableau dont chaque colonne correspond à la série
    décalée d'un lag différent.
    
    series: array 1D shape (N,)
    lags: ex. [0, 1, 2] => X(t), X(t-1), X(t-2)
    
    Retourne: array shape (N - max(lags), len(lags))
    """
    max_lag = max(lags)
    N = len(series)
    out_len = N - max_lag
    if out_len <= 0:
        raise ValueError("Nombre de points insuffisant pour supporter max_lag.")
    
    # On va construire une matrice M: shape (out_len, len(lags))
    # M[i, j] = series[i + offset - lags[j]], on doit aligner de sorte que la 1ère ligne
    # correspond à t = max_lag, la dernière à t = N-1
    data = np.zeros((out_len, len(lags)))
    for col, lag in enumerate(lags):
        # la i-ème ligne correspondra au point t = i+max_lag
        # => on veut series[t - lag]
        # => series[i+max_lag - lag]
        for i in range(out_len):
            t = i + max_lag
            data[i, col] = series[t - lag]
    return data


#############################
# TIME SERIES TE CALCULATOR
#############################

class TimeSeriesTECalculator:
    """
    Classe pour construire les data-lags et calculer la Transfer Entropy.
    On peut y ajouter un mode "fenêtre glissante".
    """
    def __init__(self, estimator: Union[KSGEstimator, KDEEstimator]):
        self.estimator = estimator

    def prepare_data_for_te(self,
                            data_dict: Dict[str, np.ndarray],
                            lags_dict: Dict[str, List[int]],
                            target_var: str,
                            target_offset: int = 1) -> Dict[str, np.ndarray]:
        """
        Construit un dictionnaire contenant :
         - 'cause': data_x (X-lags & Z-lags, etc.) => union de tous les lags "explainable variables".
         - 'condition': data_y_cond (Y-lags), s'il y a un Y-lags
         - 'target': data_y_next ( Y(t+target_offset) ), alignée temporellement.
        
        :param data_dict: ex. {'x': x_array, 'y': y_array, 'z': z_array}
        :param lags_dict: ex. {'x': [0,1,2], 'z': [0,1], 'y': [0]} => signifie qu'on inclut y(t) en condition
        :param target_var: le nom de la variable qu'on veut prédire (ex. 'y')
        :param target_offset: l'horizon de prédiction (ex. 1 => Y(t+1))
        
        Retourne un dict { 'cause': array2D, 'condition': array2D, 'target': array2D }
        alignés temporellement.
        
        Hypothèse: la cause peut inclure x(t-lags), z(t-lags), la condition inclut y(t-lags).
        """
        # 1) Construire les features "cause" en empilant les lags de x, z, etc. (sauf la target si target != x, z,...)
        # 2) Construire la condition => en général, y(t-lags). On prend lags_dict[target_var].
        # 3) Construire la target => y(t+offset)
        
        # a) On repère tous les noms de variables, et on sépare la target_var
        var_names = list(data_dict.keys())
        if target_var not in var_names:
            raise ValueError(f"La variable cible '{target_var}' n'est pas dans data_dict.")
        
        # b) On construit la "condition" => Y-lags (si y-lags existent dans lags_dict)
        cond_lags = lags_dict.get(target_var, [])
        data_y_cond = None
        if len(cond_lags) > 0:
            data_y_cond = build_lagged_features(data_dict[target_var], cond_lags)
        
        # c) On construit la cause => pour toutes variables (sauf target si on veut),
        #    mais il se peut qu'on inclue aussi la target dans la cause (ex. d'autres lags).
        #    => On suppose qu'on veut TOUTES les variables mentionnées dans lags_dict (sauf "target offset").
        cause_cols = []
        cause_dim = 0
        # On devra aligner tout le monde => on va prendre la shape minimum possible
        all_arrays = []
        for var in lags_dict:
            if var == target_var:
                # On ne l'empile pas dans cause, c'est la condition => déjà fait
                # Sauf si on veut inclure la variable target elle-même comme "cause" (?), possible en auto-reg, 
                #   Mais on suppose que c'est "condition". 
                continue
            # On fait la feature-lag
            f = build_lagged_features(data_dict[var], lags_dict[var])  # shape (M, len(lags))
            cause_cols.append(f)
        
        if len(cause_cols) == 0:
            data_cause = None
        else:
            # On doit concaténer horizontalement => Mais la longueur peut varier si max(lags) pas pareil
            # En réalité, build_lagged_features() va renvoyer un shape potentiellement différent pour chaque variable
            # (selon max_lag). On va donc devoir aligner sur la plus petite shape.
            # => On prend la min des M, on coupe le surplus.
            lengths = [c.shape[0] for c in cause_cols]
            min_len = min(lengths)
            # on trunk
            cause_cols_truncated = [c[-min_len:, :] for c in cause_cols]
            data_cause = np.hstack(cause_cols_truncated)
        
        # d) Aligner la condition data_y_cond
        if data_y_cond is not None:
            len_cond = data_y_cond.shape[0]
        else:
            len_cond = None

        # e) Construction de la target => Y(t+offset)
        # => On va créer un array shape (M, 1). M doit correspondre à la taille alignée.
        # => On prend la série data_dict[target_var], on retire max_lag du target-lags, puis on fait un offset.
        #    Simplifions : on reconstruit data_y_cond si exist. On verra la shape.
        
        # On doit connaître le max lag de y(t-lags) => max(cond_lags)
        max_lag_y = max(cond_lags) if len(cond_lags) > 0 else 0
        # => build_lagged_features a produit un array de shape (N - max_lag_y, len(cond_lags)).
        # => la 1ère ligne correspond à t = max_lag_y, la dernière à t = N-1
        # => si on veut Y(t+offset), c'est index t+offset => i-th row correspond à t,
        #    t = i + max_lag_y
        # => Y_{target}[i] = series[t + offset] = series[i + max_lag_y + offset]
        
        series_target = data_dict[target_var]
        Ntarget = len(series_target)
        M_cond = (Ntarget - max_lag_y)  # correspond à data_y_cond.shape[0] (si cond_lags>0)
        if M_cond < 1:
            raise ValueError("Not enough data for y-lags + offset.")
        
        # On va construire la target array shape (M_cond, 1)
        # target[i,0] = series_target[i + max_lag_y + offset]
        # => du coup on aura M_cond - offset lignes réellement exploitables
        # => on va créer un vecteur de taille M_cond, et on coupe la fin si besoin
        target_arr = np.zeros(M_cond)
        for i in range(M_cond):
            idx = i + max_lag_y + target_offset
            if idx >= Ntarget:
                # on ne peut plus construire la target
                # => on met un sentinel
                target_arr[i] = np.nan
            else:
                target_arr[i] = series_target[idx]
        
        # On retire toutes les lignes où target_arr[i] est NaN
        valid_mask = ~np.isnan(target_arr)
        target_arr = target_arr[valid_mask]
        target_arr = target_arr.reshape(-1, 1)

        # data_y_cond doit aussi être tronqué => data_y_cond doit avoir la même shape
        # => On coupe la fin si besoin.
        if data_y_cond is not None:
            data_y_cond = data_y_cond[valid_mask, :]
        
        # f) On fait la même chose pour data_cause => on doit l'aligner sur la dimension de data_y_cond
        # => si data_cause existe
        if data_cause is not None:
            cause_len = data_cause.shape[0]
            # On veut aligner la fin => c'est plus simple de tronquer en haut ou en bas ?
            # => On suppose qu'on a la "même" indexation : pour var \neq target, max lag = X_lags => fin similaire.
            #   la dimension min_len prise plus haut correspond à la "fenêtre" 
            #   On doit s'assurer que c'est aligné sur data_y_cond's indexing. 
            # => On fait la supposition qu'on a glissé la plus grande fenetre possible. 
            #    la dimension min_len peut être > data_y_cond.shape[0].
            min_len2 = min(data_y_cond.shape[0] if data_y_cond is not None else 999999, 
                           target_arr.shape[0],
                           data_cause.shape[0])
            data_cause = data_cause[-min_len2:, :]
            if data_y_cond is not None:
                data_y_cond = data_y_cond[-min_len2:, :]
            target_arr = target_arr[-min_len2:, :]
        else:
            # s'il n'y a pas de cause (cas pathologique), on a juste la condition + target
            min_len2 = min(data_y_cond.shape[0] if data_y_cond is not None else 999999,
                           target_arr.shape[0])
            if data_y_cond is not None:
                data_y_cond = data_y_cond[-min_len2:, :]
            target_arr = target_arr[-min_len2:, :]

        # Final
        out = {
            'cause': data_cause,     # shape (M, sum_of_lags)
            'condition': data_y_cond,  # shape (M, #y_lags)
            'target': target_arr      # shape (M, 1)
        }
        return out

    def sliding_window_te(self,
                          data_dict: Dict[str, np.ndarray],
                          lags_dict: Dict[str, List[int]],
                          target_var: str,
                          target_offset: int,
                          window_size: int) -> List[float]:
        """
        Calcule la Transfer Entropy TE(cause -> target) ~ I(cause; target | condition)
        dans des fenêtres glissantes de taille 'window_size'.
        
        :param data_dict: ex. {'x': x_array, 'y': y_array, 'z': z_array}
        :param lags_dict: ex. {'x': [0,1,2], 'z': [0,1], 'y': [0]} => Y(t) en condition
        :param target_var: ex. 'y'
        :param target_offset: horizon (ex. 1 => Y(t+1))
        :param window_size: taille de la fenêtre glissante
        :return: liste TE par fenêtre
        """
        prepared = self.prepare_data_for_te(data_dict, lags_dict, target_var, target_offset)
        data_cause = prepared['cause']
        data_cond = prepared['condition']
        data_targ = prepared['target']

        # On doit boucler sur les fenêtres, de 0 à M-window_size
        M = data_targ.shape[0]
        out_te = []
        for start in range(0, M - window_size + 1):
            end = start + window_size
            cause_win = data_cause[start:end, :] if data_cause is not None else None
            cond_win = data_cond[start:end, :] if data_cond is not None else None
            targ_win = data_targ[start:end, :]

            # TE ~ I(cause; targ_win | cond_win)
            if cause_win is None:
                # Cas bizarre : pas de cause => TE = 0 ?
                te_val = 0.0
            else:
                te_val = self.estimator.estimate_conditional_mi(
                    data_x=cause_win,
                    data_y=targ_win,
                    data_z=cond_win if cond_win is not None else np.zeros((window_size, 0))
                )
            out_te.append(te_val)
        return out_te


###########################
# DEMO
###########################
if __name__ == "__main__":

    # 1) Génération de données continues
    np.random.seed(42)
    t = np.linspace(0, 4*np.pi, 100)
    x = np.sin(t)
    y = np.sin(t + np.pi/3) + 0.1 * np.random.randn(len(t))
    z = np.cos(t) + 0.1 * np.random.randn(len(t))

    data_dict = {
        'x': x,
        'y': y,
        'z': z
    }

    # 2) Choix des lags : ex. X(t), X(t-1), Z(t), Y(t) en condition => on veut TE(X,Z -> Y(t+1) | Y(t))
    #    => On inclut y(t) dans la condition => lags_dict['y'] = [0]
    #    => On inclut x(t) et x(t-1) => lags_dict['x'] = [0,1]
    #    => On inclut z(t) => lags_dict['z'] = [0]
    lags_dict = {
        'x': [0, 1],
        'z': [0],
        'y': [0]  # Y(t) en condition
    }

    # 3) Instancier un estimateur KSG => version non-simplifiée
    ksg_est = KSGEstimator(k=5, simplified=False)

    # 4) Créer le calculateur TE
    te_calc = TimeSeriesTECalculator(estimator=ksg_est)

    # 5) Calculer TE dans des fenêtres glissantes
    #    TE({X(t),X(t-1),Z(t)} -> Y(t+1) | Y(t))
    #    window_size=10
    te_values_ksg = te_calc.sliding_window_te(
        data_dict=data_dict,
        lags_dict=lags_dict,
        target_var='y',
        target_offset=1,    # Y(t+1)
        window_size=10
    )
    print("KSG1 TE({X,Z}->Y):", te_values_ksg)

    # 6) Idem version simplifiée KSG
    ksg_est_simpl = KSGEstimator(k=5, simplified=True)
    te_calc_simpl = TimeSeriesTECalculator(estimator=ksg_est_simpl)
    te_values_simpl = te_calc_simpl.sliding_window_te(
        data_dict=data_dict,
        lags_dict=lags_dict,
        target_var='y',
        target_offset=1,
        window_size=10
    )
    print("KSG simplifié TE({X,Z}->Y):", te_values_simpl)

    # 7) KDE
    kde_est = KDEEstimator(bandwidth=0.3)
    te_calc_kde = TimeSeriesTECalculator(estimator=kde_est)
    te_values_kde = te_calc_kde.sliding_window_te(
        data_dict=data_dict,
        lags_dict=lags_dict,
        target_var='y',
        target_offset=1,
        window_size=10
    )
    print("KDE TE({X,Z}->Y):", te_values_kde)
