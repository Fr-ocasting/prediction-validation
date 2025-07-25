# Configuration des hyperparamètres
config = {
    "cnn": {
        "in_channels": 1, # On traite chaque station indépendamment au début
        "out_channels": 16,
        "kernel_size": 3,
        "padding": 1,
        "dropout": 0.2,
        'nb_quantiles' : 1
    },
    "training": {
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "epochs": 10, # Pour la démonstration
        "loss_function": "mse",
        "alpha": 0.05, # Niveau de confiance pour la calibration
        
    },
    "data": {
        "sequence_length": 6,
        "alpha_calib": 0.1, # 10% pour le set de calibration
        "test_size": 0.2,
        "valid_size": 0.2,
    }
}