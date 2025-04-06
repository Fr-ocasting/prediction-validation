import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class EntropyComputer(object):
    """
    Classe pour calculer l'entropie d'une série temporelle.
    """

    def __init__(self, nb_bins: int = 10):
        """
        :param nb_bins: Nombre de bins utilisés pour la discrétisation.
        """
        self.nb_bins = nb_bins

    def compute_entropy(self, series: np.ndarray) -> float:
        """
        Calcule l'entropie d'une série temporelle.

        :param series: Série temporelle (1D).
        :return: Entropie en nats.
        """
        # Discrétisation de la série
        hist, bin_edges = np.histogram(series, bins=self.nb_bins, density=True)
        hist = hist[hist > 0]  # Élimine les bins vides

        # Calcul de l'entropie
        entropy = -np.sum(hist * np.log(hist))
        return entropy

class TimeSeriesInfo:
    """
    Cette classe regroupe des fonctions utiles pour :
      - Discrétiser des séries temporelles.
      - Calculer l'entropie, l'entropie conjointe, et l'information mutuelle.
      - Calculer l'information mutuelle conditionnelle (Transfer Entropy simplifiée).
      - Calculer ces métriques sur des fenêtres glissantes de longueur T.
      - Gérer plusieurs séries (X1, X2, ..., Xm) pour prédire Y.
    """
    
    def __init__(self, nb_bins: int = 10):
        """
        :param nb_bins: Nombre de bins utilisés pour la discrétisation.
        """
        self.nb_bins = nb_bins

    ##############
    # DISCRETISATION
    ##############
    @staticmethod
    def digitize_series(series: np.ndarray, nb_bins: int) -> np.ndarray:
        """
        Transforme une série numérique (1D) en indices de bins.
        
        :param series: Série numérique brute.
        :param nb_bins: Nombre de bins pour l'histogramme.
        :return: Tableau d'indices entiers représentant la discrétisation.
        """
        hist, bin_edges = np.histogram(series, bins=nb_bins)
        # np.digitize renvoie l'indice du bin correspondant
        digitized = np.digitize(series, bin_edges[1:-1])
        return digitized

    ##############
    # ENTROPIES
    ##############
    @staticmethod
    def compute_entropy(discrete_values: np.ndarray, nb_bins: int) -> float:
        """
        Calcule l'entropie H(X) pour des valeurs déjà discrétisées.
        
        :param discrete_values: Indices de bins (0 à nb_bins-1 potentiellement).
        :param nb_bins: Nombre de bins.
        :return: H(X) en nats (ou bits si on change la base du log).
        """
        p, _ = np.histogram(discrete_values, bins=range(nb_bins+1), density=True)
        p = p[p > 0]  # On élimine les bins vides
        return -np.sum(p * np.log(p))
    
    @staticmethod
    def compute_joint_entropy(discrete_x: np.ndarray,
                             discrete_y: np.ndarray,
                             nb_bins: int) -> float:
        """
        Calcule l'entropie conjointe H(X, Y).
        
        :param discrete_x: Indices de bins pour X.
        :param discrete_y: Indices de bins pour Y.
        :param nb_bins: Nombre de bins.
        :return: H(X, Y).
        """
        joint_hist, _, _ = np.histogram2d(discrete_x, discrete_y,
                                          bins=[range(nb_bins+1),
                                                range(nb_bins+1)],
                                          density=True)
        joint_hist = joint_hist[joint_hist > 0]
        return -np.sum(joint_hist * np.log(joint_hist))

    @staticmethod
    def compute_conditional_entropy(discrete_target: np.ndarray,
                                   discrete_condition: np.ndarray,
                                   nb_bins: int) -> float:
        """
        Calcule l'entropie conditionnelle H(Target | Condition) = H(Target, Condition) - H(Condition).
        
        :param discrete_target: Série discrétisée (ex. Y).
        :param discrete_condition: Série discrétisée (ex. X).
        :param nb_bins: Nombre de bins.
        :return: H(Target | Condition).
        """
        # H(Target, Condition)
        joint_hist, _, _ = np.histogram2d(discrete_target, discrete_condition,
                                          bins=[range(nb_bins+1),
                                                range(nb_bins+1)],
                                          density=True)
        joint_hist = joint_hist[joint_hist > 0]
        H_target_cond = -np.sum(joint_hist * np.log(joint_hist))
        
        # H(Condition)
        p_condition, _ = np.histogram(discrete_condition, bins=range(nb_bins+1),
                                      density=True)
        p_condition = p_condition[p_condition > 0]
        H_condition = -np.sum(p_condition * np.log(p_condition))
        
        return H_target_cond - H_condition

    ##############
    # INFORMATIONS
    ##############
    def compute_mutual_information(self,
                                   discrete_x: np.ndarray,
                                   discrete_y: np.ndarray) -> float:
        """
        I(X;Y) = H(X) + H(Y) - H(X,Y)
        """
        Hx = self.compute_entropy(discrete_x, self.nb_bins)
        Hy = self.compute_entropy(discrete_y, self.nb_bins)
        Hxy = self.compute_joint_entropy(discrete_x, discrete_y, self.nb_bins)
        return Hx + Hy - Hxy
    
    def compute_conditional_mutual_information(self,
                                               discrete_x: np.ndarray,
                                               discrete_y: np.ndarray,
                                               discrete_condition: np.ndarray
                                               ) -> float:
        """
        I(X; Y | Condition) = H(X, Condition) + H(Y, Condition) - H(X, Y, Condition) - H(Condition)
        
        Ici, on simplifie souvent pour la Transfer Entropy :
        TE(X->Y) ~ I(X_t ; Y_{t+1} | Y_t)
        """
        # On va construire la distribution conjointe (X, Condition) et (Y, Condition).
        # Puis celle de (X, Y, Condition).
        
        # 1) H(X, Condition)
        joint_x_cond, _, _ = np.histogram2d(discrete_x,
                                            discrete_condition,
                                            bins=[range(self.nb_bins+1),
                                                  range(self.nb_bins+1)],
                                            density=True)
        joint_x_cond = joint_x_cond[joint_x_cond > 0]
        H_xcond = -np.sum(joint_x_cond * np.log(joint_x_cond))
        
        # 2) H(Y, Condition)
        joint_y_cond, _, _ = np.histogram2d(discrete_y,
                                            discrete_condition,
                                            bins=[range(self.nb_bins+1),
                                                  range(self.nb_bins+1)],
                                            density=True)
        joint_y_cond = joint_y_cond[joint_y_cond > 0]
        H_ycond = -np.sum(joint_y_cond * np.log(joint_y_cond))
        
        # 3) H(X, Y, Condition)
        # Pour histogrammer en 3D, on a besoin de bin3D => on combine (x, y, condition).
        # On va "empiler" [discrete_x, discrete_y, discrete_condition].
        xyz = np.vstack((discrete_x, discrete_y, discrete_condition)).T
        
        # Petites astuces pour un histogramme 3D "maison" : 
        #   on peut discretiser le vecteur "xyz" en lui-même.
        
        # Méthode "naïve" : on construit des index 3D => index = (bin_x + bin_y * nb_bins + bin_cond * nb_bins^2).
        # Puis on fait un hist 1D sur ces index.
        
        index_3d = (discrete_x
                    + discrete_y * self.nb_bins
                    + discrete_condition * self.nb_bins**2)
        
        # On normalise en densité
        hist_3d, _ = np.histogram(index_3d,
                                  bins=range(self.nb_bins**3 + 1),
                                  density=True)
        hist_3d = hist_3d[hist_3d > 0]
        H_xycond = -np.sum(hist_3d * np.log(hist_3d))
        
        # 4) H(Condition)
        p_cond, _ = np.histogram(discrete_condition,
                                 bins=range(self.nb_bins+1),
                                 density=True)
        p_cond = p_cond[p_cond > 0]
        H_cond = -np.sum(p_cond * np.log(p_cond))
        
        # I(X; Y | Condition) = H(X, Condition) + H(Y, Condition) - H(X, Y, Condition) - H(Condition)
        return H_xcond + H_ycond - H_xycond - H_cond

    ##############
    # FENETRES GLISSANTES
    ##############
    def sliding_window_mutual_info(self,
                                   series_x: np.ndarray,
                                   series_y: np.ndarray,
                                   window_size: int = 6
                                   ) -> List[float]:
        """
        Calcule I(X;Y) dans des fenêtres glissantes de taille 'window_size'.
        
        :param series_x: Série brute X(t).
        :param series_y: Série brute Y(t).
        :param window_size: Longueur de la fenêtre glissante.
        :return: Liste des I(X;Y) pour chaque fenêtre.
        """
        # On s'assure que series_x et series_y ont la même taille
        n = min(len(series_x), len(series_y))
        out = []
        
        for start in range(0, n - window_size + 1):
            end = start + window_size
            window_x = series_x[start:end]
            window_y = series_y[start:end]
            
            # Discrétisation
            dx = self.digitize_series(window_x, self.nb_bins)
            dy = self.digitize_series(window_y, self.nb_bins)
            
            # Calcul I(X;Y) sur la fenêtre
            val = self.compute_mutual_information(dx, dy)
            out.append(val)
            
        return out
    
    def sliding_window_transfer_entropy(self,
                                       series_x: np.ndarray,
                                       series_y: np.ndarray,
                                       window_size: int = 6
                                       ) -> List[float]:
        """
        Calcule une forme simplifiée de Transfer Entropy ~ I(X_t ; Y_{t+1} | Y_t)
        sur des fenêtres glissantes de taille 'window_size'.
        
        NB: On se limite ici à un décalage d'1 pas (Y_{t+1}).
            Pour un usage plus général, on décalerait series_y de 1 dans le temps.
        
        :param series_x: Série brute X(t).
        :param series_y: Série brute Y(t).
        :param window_size: Longueur de la fenêtre glissante.
        :return: Liste de TE(X->Y) pour chaque fenêtre.
        """
        n = min(len(series_x), len(series_y))
        out = []
        
        # Ici, on considère Y_{t+1}, Y_t, et X_t. 
        # -> On va "couper" la fenêtre de façon à aligner Y_{t+1} et X_t, Y_t.
        
        for start in range(0, n - window_size):
            # Fenêtre [start ... start+window_size-1] + la valeur Y_{start+window_size} pour t+1
            end = start + window_size  # on s'arrête à index end-1 inclus
            window_x = series_x[start:end]       # X(t) sur [start, end-1]
            window_y_present = series_y[start:end]  # Y(t) sur la même fenêtre
            y_next = series_y[end]  # Y_{end} = Y_{t+1}, si t = end-1
            
            # Fusionnons Y_{t+1} avec Y_t => discrete_yNext, discrete_yNow
            # On va construire un vecteur pour "Y_{t+1}" constant dans la fenêtre, 
            # mais pour la discrétisation, c'est un unique point => il faut un petit trick :
            # => On fait un "mini-tableau" de size=window_size, toujours la même valeur ?
            #    Ou on se limite au dernier point dans le calcul ?
            
            # Par simplicité, on concatène y_next sous forme de tableau d'une case
            # (un peu artificiel, mais on veut juste montrer la mécanique).
            # Dans la pratique, un calcul plus rigoureux se fait autrement.
            
            # Discrétisation X(t)
            dx = self.digitize_series(window_x, self.nb_bins)
            
            # Discrétisation Y(t)
            dy_now = self.digitize_series(window_y_present, self.nb_bins)
            
            # "Discrétisation" de Y_{t+1} en un seul point qu'on duplique
            # (juste pour avoir la même taille => window_size).
            window_y_next = np.array([y_next]*window_size)
            dy_next = self.digitize_series(window_y_next, self.nb_bins)
            
            # TE ~ I(X_t ; Y_{t+1} | Y_t)
            # => discrete_x = dx
            # => discrete_y = dy_next
            # => discrete_condition = dy_now
            te_val = self.compute_conditional_mutual_information(dx, dy_next, dy_now)
            out.append(te_val)
        
        return out

    def sliding_window_multi_transfer_entropy(self,
                                             x_series_list: List[np.ndarray],
                                             series_y: np.ndarray,
                                             window_size: int = 6
                                             ) -> List[float]:
        """
        Calcule la Transfer Entropy "groupée" : TE({X1, X2, ..., Xm} -> Y),
        c.-à-d. I({X1, ..., Xm}; Y_{t+1} | Y_t) sur une fenêtre glissante.
        
        On discrétise de manière conjointe (X1, X2, ..., Xm) en un seul indice
        (méthode un peu similaire à l'histogramme 3D généralisé).
        
        :param x_series_list: Liste de séries X1(t), X2(t), ...
        :param series_y: Série Y(t).
        :param window_size: Longueur de la fenêtre glissante.
        :return: Liste de TE sur chaque fenêtre.
        """
        n = len(series_y)
        # On suppose que toutes les X_i(t) ont au moins la même longueur (ou plus).
        for x_ser in x_series_list:
            n = min(n, len(x_ser))
        
        out = []
        for start in range(0, n - window_size):
            end = start + window_size
            
            # (a) On récupère la fenêtre glissante pour Y(t) => condition
            window_y_now = series_y[start:end]    # Y_t
            # Y_{t+1}
            y_next = series_y[end]  # Y_{end} = Y_{t+1}
            window_y_next = np.array([y_next]*window_size)
            
            # (b) Construction d'un vecteur "Xall" (X1, X2, ..., Xm) pour la fenêtre
            Xall_window = []
            for x_ser in x_series_list:
                Xall_window.append(x_ser[start:end])
            # => Xall_window est une liste de (window_size,) qu'on doit rassembler
            #    en colonnes => shape (window_size, m)
            Xall_window = np.column_stack(Xall_window)  # (T, m)
            
            # (c) Discrétisation multiple => on code (X1_bin, X2_bin, ..., Xm_bin) en un seul index
            #    Astuce : on commence par discrétiser chaque X_i individuellement.
            #    Puis on combine.
            
            # 1) Discrétiser Y(t) et Y(t+1)
            dy_now = self.digitize_series(window_y_now, self.nb_bins)
            dy_next = self.digitize_series(window_y_next, self.nb_bins)
            
            # 2) Discrétiser chaque X_i(t)
            discrete_x_cols = []
            for col_idx in range(Xall_window.shape[1]):
                col_data = Xall_window[:, col_idx]
                discrete_x_cols.append(self.digitize_series(col_data, self.nb_bins))
            # discrete_x_cols est une liste de m arrays (size=window_size).
            # On assemble en un 2D array : (window_size, m)
            discrete_x_mat = np.column_stack(discrete_x_cols)
            
            # 3) On transforme (X1, X2, ..., Xm) en un seul index (de 0 à nb_bins^m-1).
            #    On peut faire un "mix" base (nb_bins).
            #    index_agrégé(t) = X1_bin(t) + X2_bin(t)*nb_bins + ...
            # => on va faire ça de manière séquentielle
            combined_index = np.zeros(window_size, dtype=int)
            factor = 1
            for i_col in range(discrete_x_mat.shape[1]):
                combined_index += discrete_x_mat[:, i_col] * factor
                factor *= self.nb_bins
            
            # (d) TE({X1,...,Xm} -> Y) ~ I( Xall ; Y_{t+1} | Y_t )
            te_val = self.compute_conditional_mutual_information(combined_index,
                                                                 dy_next,
                                                                 dy_now)
            out.append(te_val)

        return out


###########################
# EXEMPLE D'UTILISATION
###########################
if __name__ == "__main__":
    from bokeh.plotting import figure, show, output_notebook
    from bokeh.layouts import column
    from bokeh.models import ColorBar, LinearColorMapper
    from bokeh.transform import linear_cmap
    from bokeh.io import push_notebook
    # Generation de données simples
    # Activate Bokeh in Jupyter Notebook
    output_notebook()

    # Generate simple data
    np.random.seed(42)
    t = np.linspace(0, 2 * np.pi, 50)  # 50 points
    x = np.sin(t)
    y = np.sin(t + np.pi / 3)  # phase shift
    z = np.cos(t)  # another series

    # Instantiate the class
    tsi = TimeSeriesInfo(nb_bins=10)

    # Example 1: Mutual Information (X->Y) on sliding windows
    mi_values = tsi.sliding_window_mutual_info(x, y, window_size=6)

    # Example 2: Transfer Entropy (X->Y) ~ I(X_t; Y_{t+1} | Y_t)
    te_values = tsi.sliding_window_transfer_entropy(x, y, window_size=6)

    # Example 3: Multi-series Transfer Entropy {X, Z} -> Y
    multi_te_values = tsi.sliding_window_multi_transfer_entropy([x, z], y, window_size=6)

    # Visualization with Bokeh
    def plot_entropy_values(values, title, color_palette="Viridis256"):
        """
        Helper function to plot entropy values with a colorbar.
        """
        n = len(values)
        mapper = LinearColorMapper(palette=color_palette, low=min(values), high=max(values))

        p = figure(title=title, x_axis_label="Window Index", y_axis_label="Entropy Value",
                   plot_width=800, plot_height=400)
        p.circle(x=list(range(n)), y=values, size=10, color=linear_cmap('y', color_palette, min(values), max(values)),
                 source={'x': list(range(n)), 'y': values}, legend_label="Entropy Values")
        p.line(x=list(range(n)), y=values, line_width=2, color="blue", legend_label="Trend")

        color_bar = ColorBar(color_mapper=mapper, location=(0, 0), title="Entropy")
        p.add_layout(color_bar, 'right')
        p.legend.location = "top_left"
        return p

    # Plot Mutual Information
    p1 = plot_entropy_values(mi_values, "Mutual Information (X -> Y) on Sliding Windows")

    # Plot Transfer Entropy (Single Variable)
    p2 = plot_entropy_values(te_values, "Transfer Entropy (X -> Y) on Sliding Windows")

    # Plot Multi-series Transfer Entropy
    p3 = plot_entropy_values(multi_te_values, "Multi Transfer Entropy ({X, Z} -> Y) on Sliding Windows")

    # Display all plots
    show(column(p1, p2, p3), notebook_handle=True)

    # Interpretation
    print("Interpretation:")
    print("- Mutual Information (MI): High MI values indicate a strong relationship between X and Y.")
    print("- Transfer Entropy (TE): High TE values suggest that X has a causal influence on Y.")
    print("- Multi Transfer Entropy: High values indicate that the combined influence of {X, Z} on Y is significant.")
    print("Note: These metrics do not prove causality but provide evidence of potential causal relationships.")
