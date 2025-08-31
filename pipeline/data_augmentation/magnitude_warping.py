import torch
import sys 
import os 
import pandas as pd
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

import itertools
import torch
import random

class MagnitudeWarperObject(object):
    """
    A class used to apply magnitude warping (amplitude scaling) on a specific window
    within the historical part (H) of each sequence, similarly to how the InterpolatorObject
    selects and applies interpolation windows.
    """

    def __init__(self, H, D, W, DA_magnitude_max_scale=0.2):
        """
        Args:
            H, D, W (int): same notion as in InterpolatorObject
                           total length L = W + D + H
            max_scale (float): maximum amplitude scaling factor around 1.0,
                               e.g. 0.2 => random scale in [0.8, 1.2].
        """
        self.H = H
        self.D = D
        self.W = W
        self.DA_magnitude_max_scale = DA_magnitude_max_scale

        # Détermine la zone [start_min .. end_max] correspondant à la partie H
        self.start_min = self.W + self.D
        self.end_max   = self.W + self.D + self.H - 1

        self.calendar_in_contextual = False

        self.get_all_possible_window_size_and_start()

    def get_all_possible_window_size_and_start(self):
        """
        Construit la liste de toutes les (start_idx, window_size) possibles
        dans la plage [start_min, end_max], 
        en reprenant la même logique que l'interpolation.
        """
        # start_list => indices possibles pour start
        start_list = [self.start_min + k for k in range(self.end_max - self.start_min+1)]
        # size_list => tailles de fenêtre possibles
        size_list = [k for k in range(1, (self.end_max - self.start_min))]

        possible_start_window = list(itertools.product(start_list, size_list))
        # On ne garde que ceux où (start_idx + window_size) <= end_max
        self.possible_start_window = [
            (s, w) for (s, w) in possible_start_window if s + w < self.end_max+1
        ]

    def compute_magnitude_warping(self, U_train, Utarget_train, contextual_train):
        """
        Applique une augmentation par "magnitude warping" (i.e. scaling aléatoire)
        sur les données U_train et contextual_train, en utilisant des fenêtres 
        similaires au découpage de l'interpolation.

        Args:
            U_train:  torch.Tensor de shape [n, N, L]
            Utarget_train: torch.Tensor (même shape batch, cibles)
            contextual_train: dict { name : torch.Tensor de shape [n, M, L] }

        Returns:
            (U_train_copy, Utarget_train_copy, contextual_train_copy)
        """

        # Clone original
        U_train_copy = U_train.clone()
        Utarget_train_copy = Utarget_train.clone()
        contextual_train_copy = {
            name: tensor.clone() for name, tensor in contextual_train.items()
        }

        # Shuffle the batch indices to partition them among possible windows
        n = U_train_copy.size(0)
        shuffled_index = torch.randperm(n)
        size_sub_index = n // len(self.possible_start_window) if len(self.possible_start_window) > 0 else n

        # Répartit l'index batch sur chaque (start_idx, window_size)
        for k in range(len(self.possible_start_window)):
            if k < len(self.possible_start_window) - 1:
                sub_index = shuffled_index[k * size_sub_index : (k + 1) * size_sub_index]
            else:
                sub_index = shuffled_index[k * size_sub_index : ]

            start_window = self.possible_start_window[k]
            # Applique le magnitude warping sur U_train
            U_train_copy,Utarget_train_copy = self.apply_magnitude_warping(U_train_copy,Utarget_train_copy, sub_index, start_window)
            # Idem sur contextual
            contextual_train_copy = self.apply_magnitude_warping_on_contextual_dict(contextual_train_copy, sub_index, start_window)

        if self.calendar_in_contextual:
            print('calendar data augmented by dupplication but not modified')

        return U_train_copy, Utarget_train_copy, contextual_train_copy

    def apply_magnitude_warping(self, U,Utarget, sub_index, start_window):
        """
        Applique un facteur de scaling (entre [1 - max_scale, 1 + max_scale])
        sur la fenêtre [start_idx .. end_idx] pour les échantillons sub_index.

        Args:
            U:          torch.Tensor de shape [n, N, L]
            sub_index:  indices batch à modifier
            start_window: tuple (start_idx, window_size)
        Returns:
            U modifié sur la fenêtre
        """
        start_idx, window_size = start_window
        end_idx = start_idx + window_size

        # Génère un facteur aléatoire par sample (sub_index)
        # shape = (len(sub_index),) => diffusion ensuite sur [N, (end_idx - start_idx + 1)]
        scale_factors = torch.empty(
            len(sub_index),
            device=U.device,
            dtype=U.dtype
        ).uniform_(1 - self.DA_magnitude_max_scale, 1 + self.DA_magnitude_max_scale)

        # Redimensionne pour broadcast
        # => (len(sub_index), 1, 1)
        scale_factors = scale_factors.view(-1, 1, 1)

        # Multiplie la zone [start_idx : end_idx+1]
        U[sub_index, :, start_idx : end_idx + 1] *= scale_factors

        if (end_idx == self.end_max-1) and (Utarget is not None):
            Utarget[sub_index, :, :] *= scale_factors

        return U,Utarget

    def apply_magnitude_warping_on_contextual_dict(self, contextual_train_copy, sub_index, start_window):
        """
        Parcourt le dictionnaire contextual_train_copy et applique apply_magnitude_warping
        sur chaque tensor pour 'subway_out' ou 'netmob', si c'est la même logique.
        """
        for name, tensor in contextual_train_copy.items():
            if ('subway_out' in name) or ('netmob' in name):
                tensor,_ = self.apply_magnitude_warping(tensor,None, sub_index, start_window)
            elif 'calendar' in name:
                self.calendar_in_contextual = True
            else:
                raise NotImplementedError(f'Name {name} has not been implemented for magnitude warping')

            contextual_train_copy[name] = tensor

        return contextual_train_copy
