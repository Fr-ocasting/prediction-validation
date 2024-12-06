from ray import tune

config = {#"grn_h_dim" : tune.choice([8,16,32,64,128,256]),  
        "grn_out_dim" : tune.choice([8,16,32,64,128,256]),
        }