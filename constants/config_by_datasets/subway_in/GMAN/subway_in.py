constant_name = ''
constant_modif = {
                        }

SEED = 1


config =  {'target_data': 'subway_in',
            'dataset_names': ['subway_in'],
            'dataset_for_coverage': ['subway_in'],  
            'use_target_as_context': False,
            'loss_function_type':'MSE',

            'nb_STAttblocks': 1,
            'K': 8,
            'd': 8,
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