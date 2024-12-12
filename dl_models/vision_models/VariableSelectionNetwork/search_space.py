from ray import tune

config = {#"grn_h_dim" : tune.choice([8,16,32,64,128,256]),  
        "grn_out_dim" : tune.choice([8,16,32,64,128,256]),
        "concatenation_late" : tune.choice([True,False]),
        'concatenation_early' : tune.choice([True,False]),
        }