from ray import tune

config = {#"Kt" : tune.choice([2,3,4]),  # [2,3,4]
        "stblock_num" : tune.choice([1,2,3,4]), # [2,3,4]
        #"act_func" : tune.choice(['glu']), # ['glu','gtu']
        #"Ks" :  tune.choice([3]),  #[2,3]
        #"graph_conv_type" : tune.choice(['cheb_graph_conv','graph_conv']),
        #"gso_type" : tune.choice(['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj']),
        #"adj_type" : tune.choice(['adj','corr','dist'])
        'temporal_h_dim':tune.choice([8,16,32,64,128,256]),
        'spatial_h_dim': tune.choice([8,16,32,64,128,256]),
        'output_h_dim' : tune.choice([8,16,32,64,128,256]),

        #'TGE_num_layers' : tune.choice([1,2,3,4,8]),
        #'TGE_num_heads' : tune.choice([1,2,4,8]),
        #'TGE_FC_hdim' : tune.choice([8,16,32,64,128,256]),
        }