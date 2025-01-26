import torch
import sys 
import os 
import pandas as pd
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from constants.paths import USELESS_DATES
from DL_class import FeatureVectorBuilder,DatesVerifFeatureVect
import random

class InterpolatorObject(object):
    """
    A class used for interpolation into data for augmentation.
    """

    def __init__(self, normalizers, step_ahead, H, D, W, Day_nb_steps, Week_nb_steps, shift_from_first_elmt, time_step_per_hour):
        self.normalizers = normalizers
        self.step_ahead = step_ahead
        self.H = H
        self.D = D
        self.W = W
        self.Day_nb_steps = Day_nb_steps
        self.Week_nb_steps = Week_nb_steps
        self.shift_from_first_elmt = shift_from_first_elmt
        self.time_step_per_hour = time_step_per_hour
        self.mask_seq_3d = None

        self.start_min = self.W + self.D 
        self.end_max   = self.W + self.D + self.H -1 

    def get_all_possible_window_size_and_start(self):

        start_list = [self.start_min+k for k in range(self.end_max-self.start_min)]
        size_list = [k for k in range(self.end_max-self.start_min)]

        possible_window_size = 


    def compute_interpolation(self, U_train_copy, Utarget_train_copy, ds, dataset_name, mask_inject, out_dim, alpha):
            '''
            This method retrieves a noise DataFrame for each spatial unit, reindexes it to include missing dates,
            and constructs a noise feature vector consistent with the way U_train/Utarget_train are built.
            It then applies a user-defined noise distribution (scaled and normalized) and injects the noise into
            U_train/Utarget_train to produce data-augmented tensors.

            Returns:
            --------
            U_train_copy (torch.Tensor): Augmented version of the training feature tensor
            Utarget_train_copy (torch.Tensor): Augmented version of the target feature tensor
            '''
            n, N, L = U_train_copy.shape
            # Define allowable start/end range for interpolation






            start_idx = random.randint(start_min, end_max - 1)
            end_idx   = random.randint(start_idx + 1, end_max)

            # Interpolation linéaire dans U_train_copy
            start_val = U_train_copy[:, :, start_idx]
            end_val   = U_train_copy[:, :, end_idx]
            for t in range(start_idx + 1, end_idx):
                alpha = (t - start_idx) / (end_idx - start_idx)
                U_train_copy[:, :, t] = (1 - alpha) * start_val + alpha * end_val




            self.mask_seq_3d = mask_inject.unsqueeze(-1).expand(-1, -1, L + out_dim)
            self.start = ds.tensor_limits_keeper.df_verif_train.min().min()
            self.end = ds.tensor_limits_keeper.df_verif_train.max().max()

            # 1) Retrieve and reindex noise DataFrame
            df_noises = ds.noises[dataset_name]
            noise_tensor = self._prepare_noise_tensor_from_df(df_noises)

            # 2) Build noise feature/target vectors
            U_noise, Utarget_noise = self._build_noise_feature_vectors(noise_tensor)
            
            # 3) Mask invalid dates
            amp_values, amp_values_target = self._apply_mask(U_noise, Utarget_noise, ds)

            # 4) Generate scaled noise
            scaled_noise = self._generate_scaled_noise(n, N, L, amp_values, amp_values_target, alpha,dataset_name)

            # 5) Inject noise into U_train/Utarget_train
            U_train_copy, Utarget_train_copy = self._inject_noise(
                U_train_copy, Utarget_train_copy, scaled_noise, out_dim
            )

            return U_train_copy, Utarget_train_copy

    def interpolation(self, U_train, Utarget_train, contextual_train):
        """
        A more general interpolation-based data augmentation:
        We select a random time window and linearly interpolate
        values between the start and end of that window.
        - For 'subway_out' contextual data: use one global random window for the entire batch.
        - For 'netmob' contextual data: use a different random window per sample.
        """


        # Interpolate in contextual data
        contextual_train_copy = {}
        for name, tensor in contextual_train.items():
            tensor_copy = tensor.clone()
            nbatch, nfeat, Lc = tensor_copy.shape
            
            if 'subway_out' in name:
                # Use the same global window for the entire batch
                for i in range(nbatch):
                    for j in range(nfeat):
                        start_val = tensor_copy[i, j, global_start].item()
                        end_val   = tensor_copy[i, j, global_end].item()
                        for t in range(global_start + 1, global_end):
                            alpha = (t - global_start) / (global_end - global_start)
                            tensor_copy[i, j, t] = (1 - alpha) * start_val + alpha * end_val

            elif 'netmob' in name:
                # Use a different random window for each sample
                for i in range(nbatch):
                    local_start = random.randint(start_min, end_max - 1)
                    local_end   = random.randint(local_start + 1, end_max)
                    for j in range(nfeat):
                        start_val = tensor_copy[i, j, local_start].item()
                        end_val   = tensor_copy[i, j, local_end].item()
                        for t in range(local_start + 1, local_end):
                            alpha = (t - local_start) / (local_end - local_start)
                            tensor_copy[i, j, t] = (1 - alpha) * start_val + alpha * end_val

            elif 'calendar' in name:
                print(name, 'data augmented by duplication but not modified')
            else:
                raise NotImplementedError(f'Name {name} has not been implemented for Data Augmentation')
            
            contextual_train_copy[name] = tensor_copy

        return U_train_copy, Utarget_train_copy, contextual_train_copy




def interpolation(self, U_train, Utarget_train, contextual_train):
    """
    Exemple d'interpolation se concentrant uniquement sur la partie 'H' de la séquence,
    tout en évitant d'interpoler la donnée la plus récente (index 0).
    
    Rappel du découpage supposé de la séquence de longueur L :
      - part_W : index [0..(W-1)] ou un seul point (W=1)
      - part_D : index [W..(W+D-1)] ou un seul point (D=1)
      - part_H : index [W+D .. L-1] (c'est la partie historique qu'on souhaite éventuellement interpoler)
                 NB : dans certains codes, il se peut que l'index 0 corresponde à t-1 (la donnée la plus récente),
                      ce qui renverse l'ordre. Ici, on suppose simplement qu'on veut exclure l'index 0
                      de l'interpolation, car c'est la donnée la plus récente.
    
    Hypothèse : on veut interpoler parmi les indices [W+D, ..., L-1]
                tout en évitant (L-1) si L-1 = 0 dans un schéma inversé.
                Par sécurité, on illustre comment ignorer spécifiquement l'index 0.
    """

    # Idem pour contextual_train : on applique la même logique,
    # en différenciant le comportement subway_out / netmob si nécessaire.
    contextual_train_copy = {}
    for name, tensor in contextual_train.items():
        tensor_copy = tensor.clone()
        Bc, Nc, Lc = tensor_copy.shape

        # On suppose que Lc == L. Sinon, on adapte au cas particulier.
        if Lc != L:
            print(f"Skipping interpolation for {name} (taille {Lc} != {L}).")
            contextual_train_copy[name] = tensor_copy
            continue

        if ('subway_out' in name):
            # On réutilise la même fenêtre (start_idx, end_idx) globale
            if start_min < end_max:
                for b in range(Bc):
                    for nc in range(Nc):
                        start_val = tensor_copy[b, nc, start_idx].item()
                        end_val   = tensor_copy[b, nc, end_idx].item()
                        for t in range(start_idx + 1, end_idx):
                            alpha = (t - start_idx) / (end_idx - start_idx)
                            tensor_copy[b, nc, t] = (1 - alpha) * start_val + alpha * end_val

        elif ('netmob' in name):
            # Pour netmob, on peut choisir d'avoir une fenêtre différente par sample,
            # ou de réutiliser la même par cohérence.
            # 

