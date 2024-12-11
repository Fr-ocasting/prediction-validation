from ray import tune


config = {#'calendar_class' : tune.choice([1,2,3]),
        'embedding_dim' : tune.choice([3,4,8]),
        'multi_embedding' : tune.choice([True,False]),
        #'TE_transfer' : tune.choice([True,False]),
        #"concatenation_late" : tune.choice([True,False]),
        #'concatenation_early' : tune.choice([True,False]),
        }


