from ray import tune

config = {"h_dim": tune.choice([8,16,32,64,128,256]),
          "C_outs": tune.choice([[16,32],[32,32],[64,32],[128,32],[256,32],[512,32], 
                                 [16,16],[32,16],[64,16],[128,16],[256,16],[512,16], 
          ]),
          "num_layers": tune.choice([1,2,3,4,5,6]),
          "bidirectional": tune.choice([True,False])
          }