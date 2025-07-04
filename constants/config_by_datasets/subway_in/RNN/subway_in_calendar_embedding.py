constant_name = ''
constant_modif = { }
SEED = 1

#    All Steps RMSE = 45.90, MAE = 26.63, MAPE = 29.56, MSE = 2113.09
config =  {'target_data': 'subway_in',
            'dataset_names': ['subway_in','calendar_embedding'],
            'dataset_for_coverage': ['subway_in', 'netmob_POIs'],
            'embedding_calendar_types': ['dayofweek', 'hour'],
            'use_target_as_context': False,
            'unormalize_loss' : True,
            'contextual_kwargs' : {},  
            'loss_function_type':'HuberLoss',
            'optimizer': 'adamw',
            'batch_size': 128,
            'freq': '15min',
            'standardize': False,
            'minmaxnorm': True,
            'H':6,
            'D':1,
            'W':0,

            'epochs':500,
            'input_dim' : 1,
            'h_dim' : 128, # 32 # 256
            'C_outs' : [128,32], # [32,16],# [16,16],
            'num_layers' : 4,
            'bias' : True,
            'lstm' : False,
            'gru' : False,
            'bidirectional': True,

            'TE_embedding_dim': 64,
            'TE_out_h_dim': 64,
            'TE_concatenation_late': True,
            'TE_concatenation_early':False,


            'lr': 0.001,
            'weight_decay':  0.0015,
            'dropout': 0.1,
            }