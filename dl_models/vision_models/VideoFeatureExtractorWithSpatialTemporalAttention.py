import torch
import torch.nn as nn

class VideoFeatureExtractorWithSpatialTemporalAttention(nn.Module):
    def __init__(self, c_in=3, out_dim=64, N=40, d_model=128):
        super(VideoFeatureExtractorWithSpatialTemporalAttention, self).__init__()
        self.z_dim = out_dim
        self.N = N
        self.d_model = d_model

        # Encodeur convolutionnel
        self.encoder = nn.Sequential(
            nn.MaxPool3d((2,2,1)),  # a voir si on conserve ça 
            nn.Conv3d(c_in, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, H/2, W/2, L/2]
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, d_model, kernel_size=3, stride=2, padding=1),  # [B, d_model, H/4, W/4, L/4]
            nn.BatchNorm3d(d_model),
            nn.ReLU(inplace=True),
        )

        # Embeddings de requête pour N stations
        self.query_embed = nn.Parameter(torch.randn(N, self.z_dim))

        # Projection des caractéristiques extraites
        self.key_proj = nn.Conv3d(d_model, self.z_dim, kernel_size=1)
        self.value_proj = nn.Conv3d(d_model, self.z_dim, kernel_size=1)

        # Mécanisme d'attention
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)  # [B, d_model, H', W', L']
        C, H, W, L = x.size(1), x.size(2), x.size(3), x.size(4)
        S = H * W * L

        # Projeter les caractéristiques en clés et valeurs
        keys = self.key_proj(x)    # [B, z_dim, H', W', L']
        values = self.value_proj(x)  # [B, z_dim, H', W', L']

        # Aplatir les dimensions spatiales et temporelles
        keys = keys.view(B, self.z_dim, -1)      # [B, z_dim, S]
        values = values.view(B, self.z_dim, -1)  # [B, z_dim, S]

        # Normalisation des clés
        keys = keys / (self.z_dim ** 0.5)

        # Préparer les requêtes pour chaque station
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, N, z_dim]

        # Calcul de l'attention
        attention_scores = torch.bmm(queries, keys)  # [B, N, S]
        attention_weights = self.softmax(attention_scores)  # [B, N, S]

        # Calcul des représentations latentes pour chaque station
        context = torch.bmm(attention_weights, values.transpose(1, 2))  # [B, N, z_dim]

        context = context.reshape(context.size(0),-1) # [B, N *z_dim]
        return context  