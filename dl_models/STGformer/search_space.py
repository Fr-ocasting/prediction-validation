from ray import tune

config = {
    # Dimensions d'embedding
    "input_embedding_dim": tune.choice([16, 24, 32, 48, 64]),
    "tod_embedding_dim": tune.choice([0, 4, 8, 12, 16]),
    "dow_embedding_dim": tune.choice([0, 4, 8, 12, 16]),
    "spatial_embedding_dim": tune.choice([0, 8, 16, 32]),
    "adaptive_embedding_dim": tune.choice([0, 8, 12, 16, 24]),

    # Attention
    "num_heads": tune.choice([1, 2, 4, 8]),
    "num_layers": tune.choice([1, 2, 3, 4, 6]),
    "mlp_ratio": tune.choice([1.0, 1.5, 2.0, 2.5, 3.0]),  # PAS SUR 

    # Projections et dropout
    "use_mixed_proj": tune.choice([False, True]),
    "dropout_a": tune.uniform(0.0, 0.5),

    # Kernel sizes pour la projection temporelle
    "kernel_size": tune.choice([[1], [3], [1, 3], [3, 5]]),

    # gso:
    'adj_normalize_method':tune.choice(['normlap','scalap','symadj','transition','doubletransition','identity']) 

}