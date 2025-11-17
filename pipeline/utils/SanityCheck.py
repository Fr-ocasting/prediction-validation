import torch
import torch.nn as nn
from torch import Tensor

class SanityCheck:
    """
    Classe utilitaire pour déboguer les tenseurs à des points critiques.
    Initialiser avec active=True pour l'entraînement/débogage, active=False pour l'inférence.
    """
    def __init__(self, active=True,grad_dict=None ):
        self.active = active
        self.counter = 0
        self.grad_dict = grad_dict
        print(f"SanityCheck initialisé. Statut: {'ACTIF' if active else 'INACTIF'}")

    def checking(self, x, name="", dim_to_check=None, module_name = ''):
        """
        Imprime les statistiques d'un tenseur.
        :param x: Le tenseur à vérifier.
        :param name: Un nom descriptif pour ce point de contrôle.
        :param dim_to_check: (Optionnel) Une dimension spécifique pour calculer l'écart-type
                             (utile pour vérifier l'effondrement des 'nodes' ou des 'tokens').
        """
        if not self.active:
            return

        if not isinstance(x, torch.Tensor):
            print(f"\n--- [SANITY CHECK {self.counter}: {module_name}] ---")
            print(f"  {name}] ---")
            print(f"  ERREUR: Objet n'est pas un tenseur. Type: {type(x)}")
            print("---------------------------------")
            self.counter += 1
            return

        with torch.no_grad():
            stats_x = x.detach().cpu().float()

            if stats_x.numel() == 0:
                print(f"\n--- [SANITY CHECK {self.counter}: {module_name}] ---")
                print(f"  {name}] ---")
                print(f"  Shape: {stats_x.shape} (TENSEUR VIDE)")
                print("---------------------------------")
                self.counter += 1
                return

            # Statistiques globales
            mean_val = stats_x.mean().item()
            std_val = stats_x.std().item()
            min_val = stats_x.min().item()
            max_val = stats_x.max().item()

            # Problèmes courants
            num_nans = torch.isnan(stats_x).sum().item()
            num_infs = torch.isinf(stats_x).sum().item()
            num_zeros = (stats_x == 0).sum().item()
            total_elements = stats_x.numel()

            pct_nans = (num_nans / total_elements) * 100
            pct_infs = (num_infs / total_elements) * 100
            pct_zeros = (num_zeros / total_elements) * 100

            # --- Affichage ---
            print(f"\n--- [SANITY CHECK {self.counter}: {module_name}] ---")
            print(f"  {name}] --- Shape: {stats_x.shape}")
            print(f"  Mean: {mean_val:.4e} | Std: {std_val:.4e}")
            print(f"  Min:  {min_val:.4e} | Max: {max_val:.4e} | Zeros: {pct_zeros:.2f}% | NaNs: {pct_nans:.2f}% | Infs: {pct_infs:.2f}%")

            # Vérification de l'écart-type sur une dimension spécifique
            if dim_to_check is not None:
                try:
                    if stats_x.dim() > dim_to_check:
                        # Calcule l'écart-type sur la dimension (ex: les 40 stations)
                        std_across_dim = stats_x.std(dim=dim_to_check, unbiased=False)
                        # Affiche la moyenne de ces écarts-types
                        print(f"  Std across dim={dim_to_check} (Mean): {std_across_dim.mean().item():.4e}")
                    else:
                        print(f"  Std Check: dim {dim_to_check} hors limites pour shape {stats_x.shape}")
                except Exception as e:
                    print(f"  Std Check: Erreur lors du calcul std sur dim={dim_to_check}: {e}")
            


        if self.grad_dict is not None and module_name:
            print(f"  --- Tracked Gradients (from module '{module_name}') ---")
            found_grad = False
            
            # Trie les clés pour un affichage cohérent
            sorted_grad_keys = sorted(self.grad_dict.keys())

            for grad_name in sorted_grad_keys:
                if module_name in grad_name:
                    param_name = grad_name.replace(module_name, '', 1).lstrip('/')
                    norm_value_list = self.grad_dict[grad_name]
                    if isinstance(norm_value_list, list) and len(norm_value_list) > 0:
                        norm_value = norm_value_list[-1]
                        found_grad = True
                        print(f"    {param_name:<20} | Norm: {norm_value:.4e}")
                    
            if not found_grad:
                print("    No tracked gradients for this module.")
        elif self.grad_dict is None and module_name and self.active:
             print("  --- Activated gradient tracking but no gradient dictionary provided. ---")

        print("---------------------------------")
        
        self.counter += 1

