from ray import tune
from itertools import product

config = {'nb_block' : tune.choice([1,2,3,4]),
            'K' : tune.choice([1,2,3]),
            'nb_chev_filter' : tune.choice([16,32,64,128,256]),
            'nb_time_filter' : tune.choice([16,32,64,128,256]),
            'threshold' : tune.choice([[0.1,0.3,0.7]])
          }
