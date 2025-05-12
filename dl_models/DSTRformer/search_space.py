from ray import tune

config = {
    "input_embedding_dim": tune.choice([8,16, 24]),
    "tod_embedding_dim": tune.choice([4, 8]),
    "dow_embedding_dim": tune.choice([4, 8]),
    "adaptive_embedding_dim": tune.choice([8, 12, 16, 24, 32]),
    "node_dim": tune.choice([8, 12, 16, 24, 32]),
    "out_feed_forward_dim": tune.choice([8, 16, 32, 64,128]),
    #"num_layers_m": tune.randint(1,6),
    "ts_embedding_dim": tune.choice([4, 8, 12, 16]),
    #"time_embedding_dim": tune.choice([0, 4, 8, 12, 16]),
    #"mlp_num_layers": tune.randint(1,6),
    
    # Attention
    "num_heads": tune.choice([1, 2, 3]), #has to devide input_embedding_dim+tod_embedding_dim+dow_embedding_dim+adaptive_embedding_dim+spatial_embedding_dim
    "num_layers": tune.randint(1,3),
    "feed_forward_dim": tune.choice([8, 16, 32, 64, 128, 256]),
    #"use_mixed_proj": tune.choice([True, False]),
}