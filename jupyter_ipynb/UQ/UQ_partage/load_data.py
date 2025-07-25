import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def create_sequences(data: pd.DataFrame, sequence_length: int):
    """Crée des séquences de données et les cibles correspondantes."""
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data.iloc[i:i + sequence_length].values)
        targets.append(data.iloc[i + sequence_length].values)
    
    # Convertir en Tensors
    # X: (nombre_sequences, L, N)
    # y: (nombre_sequences, N)
    X = torch.tensor(np.array(sequences), dtype=torch.float32).permute(0, 2, 1)
    y = torch.tensor(np.array(targets), dtype=torch.float32)
    print(f"Dimensions des séquences: X={X.shape}, y={y.shape}")
    
    return X, y

def get_dataloaders(X, y, config):
    """Divise les données et crée les DataLoaders PyTorch."""
    
    # Séparation Test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"], shuffle=False
    )
    
    # Séparation Train et Validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_val, y_train_val, test_size=config["data"]["valid_size"], shuffle=False
    )
    
    # Séparation Proper Train et Calibration sets
    X_proper_train, X_calib, y_proper_train, y_calib = train_test_split(
        X_train, y_train, test_size=config["data"]["alpha_calib"], shuffle=False
    )
    
    # Création des TensorDatasets
    train_dataset = TensorDataset(X_proper_train, y_proper_train)
    calib_dataset = TensorDataset(X_calib, y_calib)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Création des DataLoaders
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print('Batch size:', batch_size)
    print(f"Taille des Set:  Proper-Train / Calib-Train / Valid / Test : {len(train_loader),len(calib_loader),len(valid_loader),len(test_loader)} Batches")
    return {
        "train": train_loader,
        "calib": calib_loader,
        "valid": valid_loader,
        "test": test_loader
    }