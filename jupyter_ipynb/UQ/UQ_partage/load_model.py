
import torch
import torch.nn as nn
import numpy as np
from plotting import plot_uncertainty_bands


class DeepEnsemble(object):
    def __init__(self,Trainers):
        super(DeepEnsemble,self).__init__()
        self.Trainers = Trainers

    def train_and_test_n_times(self):
        self.L_predictions = []
        for trainer in self.Trainers: 
            trainer.train_and_valid()
            test_predictions,real_values = trainer.predict(mode='test')
            self.L_predictions.append(test_predictions.squeeze())

        self.real_values =real_values.squeeze()

    def plot_spatial_unit_i(self,station_i, window_pred = np.arange(2*96), method = 'std_range', Lambda_coeffs =[1,2,3]):
        plot_uncertainty_bands(self.L_predictions,self.real_values,station_i, window_pred = window_pred, method = method, Lambda_coeffs =Lambda_coeffs)





class CNN(nn.Module):
    def __init__(self, num_nodes, seq_len, cnn_config,nb_quantiles = 1):
        """
        Args:
            num_nodes (int): Nombre de stations de métro (N).
            seq_len (int): Longueur de la séquence d'entrée (L).
            cnn_config (dict): Dictionnaire de configuration pour le CNN.
        """
        super(CNN, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        
        # Le CNN traitera la dimension temporelle (L)
        self.conv1 = nn.Conv1d(
            in_channels=num_nodes,
            out_channels=cnn_config["out_channels"],
            kernel_size=cnn_config["kernel_size"],
            padding=cnn_config["padding"],
            groups=1 # Chaque "noeud" est traité par des filtres différents au début
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(cnn_config["dropout"])
        
        # Couche de sortie pour prédire le trafic à t+1 pour chaque station
        # La taille de sortie de la conv1d est (B, out_channels, L)
        self.fc = nn.Linear(cnn_config["out_channels"] * seq_len, num_nodes*nb_quantiles)

    def forward(self, x):
        # x: (B, N, L)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Aplatir la sortie pour la couche linéaire
        # (B, out_channels, L) -> (B, out_channels * L)
        x = x.view(x.size(0), -1)
        
        # Prédire la sortie pour chaque noeud
        # (B, N*nb_quantiles)
        output = self.fc(x)
        # Reshape pour avoir (B, N, nb_quantiles)
        output = output.view(x.size(0), self.num_nodes, -1)

        return output
