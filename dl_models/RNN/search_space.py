from ray import tune

config = {"h_dim": tune.choice([8,16,32,64,128]),
          "C_outs": tune.choice([[16,1],[32,1],[64,1],[128,1],[16, 8,1],[32,16,1],[32,32,1]]),
          "num_layers": tune.choice([1,2,3,4]),
          "bidirectional": tune.choice([True,False])
          }