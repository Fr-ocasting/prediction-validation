constant_name = ''
constant_modif = {
                        }

SEED = 1


config =  {'target_data': 'subway_in',
            'dataset_names': ['subway_in'],
            'dataset_for_coverage': ['subway_in'],  
            'use_target_as_context': False,
            'loss_function_type':'MSE',

            # GMAN parameters:
            'nb_STAttblocks': 1,
            'num_heads': 8,  # 'num_heads = K = 8' in the paper
            'head_dim': 8,   # 'head_dim = d = 8' in the paper
            'bn_decay': 0.1,
            'adj_type': 'dist',


            'batch_size': 32,
            'lr': 0.001,
            'dropout': 0.0,
            'epochs':100, #200, 
            'standardize': False,
            'minmaxnorm': True,
            'unormalize_loss' : True,


            'optimizer': 'adam',
            'freq': '15min',
            'H':6,
            'D':1,
            'W':0,
              }