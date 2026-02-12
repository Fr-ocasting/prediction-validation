from ray import tune

config = {"layers" : tune.choice([1,2,3,4]),
          "end_channels":  tune.choice([16,32,64,128]),
          "skip_channels":  tune.choice([16,32,64,128]),
        "conv_channels":  tune.choice([16,32,64,128]),          
         "residual_channels":  tune.choice([16,32,64,128]),                
          "dilation_exponential":  tune.choice([1,2,3]),  
        }
