from ray import tune

config = {
    "input_embedding_dim": tune.choice([8,16, 24, 32, 48, 64]),
    #"tod_embedding_dim": tune.choice([0, 4, 8, 12, 16]),
    #"dow_embedding_dim": tune.choice([0, 4, 8, 12, 16]),
    "adaptive_embedding_dim": tune.choice([8, 12, 16, 24, 32]),
    
    # Attention
    "num_heads": tune.choice([1, 2, 4]), #has to devide input_embedding_dim+tod_embedding_dim+dow_embedding_dim+adaptive_embedding_dim
    "num_layers": tune.randint(1,6),
    "mlp_ratio": tune.uniform(1, 4).quantized(0.2),
    
    # Adaptive embedding dropout
    "dropout_a": tune.uniform(0.0, 0.5),
    
    # Kernel sizes for temporal projection
    "kernel_size": tune.choice([[1], [1,1], [1, 3], [3, 3]]),
}