constant_name = ''
constant_modif = { }
SEED = 1
config =  {'target_data': 'subway_in',
            'dataset_names': ['subway_in','calendar'],
            'dataset_for_coverage': ['subway_in'],
            'use_target_as_context': False,
            'input_embedding_dim': 12,
            'unormalize_loss' : True,
            'contextual_kwargs' : {},  
            'loss_function_type':'HuberLoss',
            'optimizer': 'adam',
            'batch_size': 128,
            'epochs':200, # 500, 

            'adaptive_embedding_dim': 80,
            'input_embedding_dim': 24,
            'tod_embedding_dim': 0,
            'dow_embedding_dim': 0,
            'feed_forward_dim': 256,
            'num_heads': 4,
            'num_layers': 3,

            'use_mixed_proj': True,
            'freq': '15min',
            'H':6,
            'D':1,
            'W':0,

            'lr': 0.001,
            'weight_decay':  0.0003,
            'dropout': 0.1,

            'torch_scheduler_milestone': [20,30],
            'torch_scheduler_gamma':0.1,
            'torch_scheduler_type': 'MultiStepLR',

            'standardize': False,
            'minmaxnorm': True,
            'calendar_types':['dayofweek', 'timeofday']
                            }