constant_name = ''
constant_modif = {}
SEED = 1


config =  {'target_data': 'subway_in',
            'dataset_names': ['subway_in', 'calendar_embedding'],
            'dataset_for_coverage': ['subway_in'],  # ['subway_in', 'netmob_POIs'], 
            'embedding_calendar_types': ['dayofweek', 'hour'],
            'use_target_as_context': False,
            'loss_function_type':'MSE',
            'Kt': 3,
            'stblock_num': 2,
            'Ks': 3,
            'graph_conv_type': 'graph_conv',
            'gso_type': 'sym_renorm_adj',
            'enable_bias': True,
            'adj_type': 'corr',
            'enable_padding': True,
            'threshold': 0.3,
            'act_func': 'glu',
            'temporal_h_dim': 64,
            'spatial_h_dim': 16,
            'output_h_dim': 128,
            'weight_decay': 0.001,
            'batch_size': 128,
            'lr': 0.001,
            'dropout': 0.0,
            'epochs':500,
            'standardize': False,
            'minmaxnorm': True,
            'unormalize_loss' : True,

            'TE_embedding_dim': 64,
            'TE_out_h_dim': 64,
            'TE_concatenation_late': True,
            'TE_concatenation_early':False,

            'optimizer': 'adamw',
            'freq': '15min',
            'H':6,
            'D':1,
            'W':0,
              }