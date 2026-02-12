import torch
import numpy as np
import argparse
import copy

# Assurez-vous que les fichiers GMAN_orig.py et GMAN.py sont dans le même dossier
from GMAN_orig import GMAN as GMAN_Original
from GMAN import GMAN as GMAN_Modified

def verify():
    # 1. Setup Global
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Dimensions
    B, N = 2, 5       # Batch, Nodes
    Seq_Len = 12      # Longueur historique (num_his)
    T_pred = 12       # Longueur prédiction
    Nb_Layers = 3     # Nombre de blocs STAttBlock (L dans l'original)
    C = 1             # Channels
    K, d = 2, 4       # Heads, Dim per head
    D = K * d
    
    # 2. Préparation des Données Identiques
    # X_common: [Batch, Seq_Len, Num_Nodes]
    X_common = torch.randn(B, Seq_Len, N)
    
    # TE_common: [Batch, Total_Steps, 2]
    TE_common = torch.randint(0, 7, (B, Seq_Len + T_pred, 2)).float()
    
    # SE_common: [Num_Nodes, D]
    SE_common = torch.randn(N, D)

    # ---------------------------------------------------------
    # 3. Modèle ORIGINAL
    # ---------------------------------------------------------
    parser_orig = argparse.ArgumentParser()
    args_orig = parser_orig.parse_args()
    
    # ATTENTION: Dans l'original, 'L' signifie 'Number of Layers'
    args_orig.L = Nb_Layers       
    # 'num_his' signifie 'History Steps'
    args_orig.num_his = Seq_Len   
    
    args_orig.K = K
    args_orig.d = d
    args_orig.bn_decay = 0.1
    
    # Init Original
    model_orig = GMAN_Original(SE_common, args_orig, bn_decay=0.1)
    model_orig.eval()

    # ---------------------------------------------------------
    # 4. Modèle MODIFIÉ (Framework)
    # ---------------------------------------------------------
    parser_mod = argparse.ArgumentParser()
    args_mod = parser_mod.parse_args()
    
    # Mapping des arguments
    args_mod.num_nodes = N
    args_mod.L = Seq_Len          # Ici L est la séquence historique
    args_mod.out_steps = T_pred
    args_mod.step_ahead = T_pred
    args_mod.horizon_step = 1
    args_mod.C = C
    args_mod.bn_decay = 0.1
    args_mod.num_layers = Nb_Layers # On s'assure que c'est bien 3
    args_mod.nb_STAttblocks = Nb_Layers
    args_mod.num_heads = K
    args_mod.head_dim = d
    args_mod.steps_per_day = 288
    args_mod.time_step_per_hour = 288//24

    
    # Args contextuels vides
    args_mod.contextual_kwargs = {}
    args_mod.contextual_positions = {}
    args_mod.Early_fusion_names = []
    args_mod.Late_fusion_names = []

    # Init Modifié
    model_mod = GMAN_Modified(args_mod, SE=SE_common.clone())
    model_mod.eval()

    # ---------------------------------------------------------
    # 5. TRANSFERT DES POIDS
    # ---------------------------------------------------------
    print(f"Configuration: {Nb_Layers} Layers, Seq_Len={Seq_Len}")
    print("Transfert des poids...")
    
    state_dict = model_orig.state_dict()
    missing, unexpected = model_mod.load_state_dict(state_dict, strict=False)
    
    # Transfert manuel du SE qui n'est pas dans le state_dict de l'original (car attribut simple)
    # mais qui est un Parameter dans le modifié
    model_mod.SE.data = model_orig.SE.data.clone()

    # ---------------------------------------------------------
    # 6. Comparaison
    # ---------------------------------------------------------
    
    # --- Forward Original ---
    with torch.no_grad():
        out_orig = model_orig(X_common, TE_common) # [B, T_pred, N]

    # --- Forward Modifié ---
    # Adaptation Input: [B, Seq_Len, N] -> [B, 1, N, Seq_Len] -> [B, C, N, Seq_Len]
    x_input_mod = X_common.unsqueeze(1).permute(0, 1, 3, 2)
    
    # Adaptation Calendar: [B, Tot, 2] -> [B, 2, N, Tot]
    x_cal_input_mod = TE_common.unsqueeze(2).repeat(1, 1, N, 1).permute(0, 3, 2, 1)

    with torch.no_grad():
        out_mod = model_mod(x_input_mod, x_calendar=x_cal_input_mod) # [B, 1, N, T_pred]
    
    # Alignement output modifié pour comparaison: [B, 1, N, T] -> [B, T, N]
    out_mod_aligned = out_mod.squeeze(1).permute(0, 2, 1)

    # ---------------------------------------------------------
    # 7. RÉSULTATS
    # ---------------------------------------------------------
    diff = torch.abs(out_orig - out_mod_aligned).max().item()
    print(f"\nDifférence Maximale: {diff}")
    
    if diff < 1e-5:
        print("✅ SUCCÈS : Les modèles sont équivalents.")
    else:
        print("❌ ÉCHEC : Les sorties diffèrent encore.")

if __name__ == '__main__':
    verify()