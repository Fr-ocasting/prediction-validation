from ray import tune

config = {"max_diffusion_step" : tune.choice([1,2,3,4]),
        "adj_type" : tune.choice(['adj','corr','dist']),
        "filter_type": tune.choice(['laplacian', 'random_walk', 'dual_random_walk']),
        "num_rnn_layers": tune.choice([1,2,3,4]),
        "rnn_units": tune.choice([1,2,3,4])
        }