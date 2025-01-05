from ray import tune

config = {#"grn_h_dim" : tune.choice([8,16,32,64,128,256]),  
        #"grn_out_dim" : tune.choice([4,8,16,32,64]),
        #"concatenation_late" : tune.choice([True,False]),
        #'concatenation_early' : tune.choice([True,False]),
        'concatenation_order': tune.choice([{"concatenation_early" :True, "concatenation_late" :True},
                                        {"concatenation_early" :True, "concatenation_late" :False},
                                        {"concatenation_early" :False, "concatenation_late" :True},
                                        ]),
        'n_head_d_model': tune.choice([{'num_heads' :1, "grn_out_dim" :tune.choice([4,8,16,32,64])},
                                        {'num_heads' :2, "grn_out_dim" :tune.choice([8,16,32,64,128])},
                                        {'num_heads' :3, "grn_out_dim" :tune.choice([12,24,48,96,192])},
                                        {'num_heads' :4, "grn_out_dim" :tune.choice([16,32,64,128,256])},
                                        {'num_heads' :6, "grn_out_dim" :tune.choice([24,48,96,192])},
                                        ]),
        }