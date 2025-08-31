import torch
import torch.nn as nn

class FeatureExtractorEncoderDecoder(nn.Module):
    def __init__(self, c_in=3, out_dim=64, N=40,H=268,W=287,L=8):
        super(FeatureExtractorEncoderDecoder, self).__init__()
        self.z_dim = out_dim
        self.num_nodes = num_nodes

        # Encodeur
        self.encoder = nn.Sequential(
            nn.MaxPool3d((2,2,1)),
            nn.Conv3d(c_in, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, H/2, W/2, L/2]
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),    # [B, 64, H/4, W/4, L/4]
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),   # [B, 128, H/8, W/8, L/8]
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        # Calcul de la taille après les convolutions pour le flatten
        self._initialize_flatten_size(c_in,H,W,L)

        # Couche entièrement connectée pour projeter dans l'espace latent
        self.fc = nn.Linear(self.flatten_size, self.z_dim * N)

    def _initialize_flatten_size(self,c_in,H,W,L):
        # On crée un tenseur factice pour calculer la taille après les convolutions
        with torch.no_grad():
            x = torch.zeros(1, c_in, H, W, L)  # Taille d'entrée
            x = self.encoder(x)
            self.flatten_size = x.view(1, -1).size(1)
    
    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)  # [B, C, H', W', L']
        x = x.view(B, -1)    # [B, flatten_size]
        x = self.fc(x)       # [B, z_dim * N]
        return x