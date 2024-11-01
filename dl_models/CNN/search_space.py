from ray import tune
from itertools import product

h1 = [8,16,32,64,128]
h2 = [8,16,32,64,128]
possible_H_dims = [[a,b] for (a,b) in list(product(h1,h2))]

c1 = [8,16,32,64,128]
c2 = [1]
possible_C_outs = [[a,b] for (a,b) in list(product(c1,c2))]

config = {"H_dims": tune.choice(possible_H_dims),
          "C_outs": tune.choice(possible_C_outs)
          }