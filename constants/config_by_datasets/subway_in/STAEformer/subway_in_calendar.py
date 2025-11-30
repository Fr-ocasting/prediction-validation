constant_name = ''
constant_modif = { }
SEED = 1
config = {'target_data': 'subway_in',
         'dataset_names': ['subway_in','calendar'],
         'dataset_for_coverage': ['subway_in'], # ['subway_in', 'netmob_POIs'],
        'use_target_as_context': False,
        'unormalize_loss' : True,
        'contextual_kwargs' : {},  
        'loss_function_type':'HuberLoss',
        'optimizer': 'adamw',
        'batch_size': 128,
        'epochs':200, # 500, 
        'adaptive_embedding_dim': 32,
        'input_embedding_dim': 24,
        'tod_embedding_dim': 6,
        'dow_embedding_dim': 6,
        'feed_forward_dim': 256,
        
        'num_heads': 4,
        'num_layers': 3,

        'use_mixed_proj': True,
        'freq': '15min',
        'H':6,
        'D':1,
        'W':0,

        'lr': 0.001,
        'weight_decay':  0.0015,
        'dropout': 0.2,
        'torch_scheduler_milestone': 20,
        'torch_scheduler_gamma':0.9925,
        'torch_scheduler_type': 'warmup',
        'torch_scheduler_lr_start_factor': 0.3,
        'standardize': True,
        'minmaxnorm': False,
        'calendar_types':['dayofweek', 'timeofday'],
                            }