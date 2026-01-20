from ray import tune

config = {'num_heads' : tune.choice([4,8]),
          'head_dim' : tune.choice([8,16,32]),
          'nb_STAttblocks' : tune.choice([2,3])
        }