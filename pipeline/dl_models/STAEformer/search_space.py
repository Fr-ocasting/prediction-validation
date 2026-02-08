from ray import tune

config = {
    "input_embedding_dim": tune.choice([16,32,64]),
    "tod_embedding_dim": tune.choice([4,12,24]),
    "dow_embedding_dim": tune.choice([4,12,24]),
    "adaptive_embedding_dim": tune.choice([16,32,64]),
    # # "spatial_embedding_dim": tune.choice([4, 8, 12, 16,32]),
    
    # # Attention
    # "num_heads": tune.choice([1, 2, 4]), #has to devide input_embedding_dim+tod_embedding_dim+dow_embedding_dim+adaptive_embedding_dim+spatial_embedding_dim
    # "num_layers": tune.choice([2, 3, 4]),
    # "feed_forward_dim": tune.choice([8, 16, 32,64,128,256,512]),
    # #"use_mixed_proj": tune.choice([True, False]),
}