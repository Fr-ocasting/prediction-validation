constant_name = ''
constant_modif = {}
SEED = 1


config =  {'target_data': 'subway_in',
            'dataset_names': ['subway_in', 'calendar'],
            'dataset_for_coverage': ['subway_in'],  
            'use_target_as_context': False,
            'loss_function_type':'HuberLoss',

            # GMAN parameters:
            'nb_STAttblocks': 2,
            'num_heads': 4,  # = k = 8 in the paper
            'head_dim': 16,  # = d = 8 in the paper
            'bn_decay': 0.1,
            'adj_type': 'dist',
            # ---

            # Training / Optim parameters:
            'lr': 0.0030556064730659,
            'weight_decay': 0.0036466444276375,
            'dropout': 0.0,   # 0.145169206052754,  Does not exist in the GMAN
            'batch_size': 128,
            'epochs':100, #200, 
            'torch_scheduler': None,
            # ---

            'standardize': False,
            'minmaxnorm': True,
            'unormalize_loss' : True,

            'optimizer': 'adamw',
            'freq': '15min',
            'H':6,
            'D':1,
            'W':0,
                
            'calendar_types':['dayofweek', 'timeofday'],

              }




