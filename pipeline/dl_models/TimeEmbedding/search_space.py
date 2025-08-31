from ray import tune


config = {#'calendar_class' : tune.choice([1,2,3]),
        #'embedding_dim' : tune.choice([3,4]),
        #'multi_embedding' : tune.choice([True,False]),
        #'TE_transfer' : tune.choice([True,False]),

        ## === Set the possibility if 'calendar' is on: concatenation early or concatenation late.
        #"concatenation_late" : tune.choice([True,False]),
        #'concatenation_early' : tune.choice([True,False]),
        #'concatenation_order': tune.choice([{"concatenation_early" :True, "concatenation_late" :True},
        #                                     {"concatenation_early" :True, "concatenation_late" :False},
        #                                     {"concatenation_early" :False, "concatenation_late" :True},
        #                                    ]),
        ## ====
        #'fc1' : tune.choice([{'fc1':tune.choice([4,8,16,32,64]),'fc2':tune.choice([4,8,16,32,64]),'activation_fc1': tune.choice([True,False])},
        #                     {'fc1':tune.choice([4,8,16,32,64]),'fc2':None,'activation_fc1':None}
        #                    ]),
        #'out_h_dim' : tune.choice([4,8,16,32]),
        
        }

