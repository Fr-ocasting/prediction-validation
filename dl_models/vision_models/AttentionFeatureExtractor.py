import torch
import torch.nn as nn

class AttentionFeatureExtractor(nn.Module):
    def __init__(self, c_in=3, out_dim=64, n_vertex=40):
        super(AttentionFeatureExtractor, self).__init__()
        self.z_dim = out_dim
        self.n_vertex = n_vertex

        # Encodeur convolutionnel
        self.encoder = nn.Sequential(
            nn.MaxPool3d((2,2,1)),
            nn.Conv3d(c_in, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Projections pour les clés, valeurs et requêtes
        self.key_conv = nn.Conv3d(128, self.z_dim, kernel_size=1)
        self.value_conv = nn.Conv3d(128, self.z_dim, kernel_size=1)
        self.query_embed = nn.Parameter(torch.randn(N, self.z_dim))

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)  # [B, 128, H', W', L']
        H, W, L = x.size(2), x.size(3), x.size(4)
        S = H * W * L

        keys = self.key_conv(x).view(B, self.z_dim, -1)      # [B, z_dim, S]
        values = self.value_conv(x).view(B, self.z_dim, -1)  # [B, z_dim, S]
        queries = self.query_embed.unsqueeze(0).repeat(B, 1, 1)  # [B, N, z_dim]

        # Calcul de l'attention
        attention_scores = torch.bmm(queries, keys) / (self.z_dim ** 0.5)  # [B, N, S]
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)  # [B, N, S]

        # Calcul des représentations des stations
        x_latent = torch.bmm(attention_weights, values.transpose(1, 2))  # [B, N, z_dim]
        x_latent = x_latent.transpose(1, 2)  # [B, z_dim, N]
        x_latent = x_latent.view(x_latent.size(0),-1)  # [B, z_dim, N] -> # [B, N*z_dim]

        return x_latent