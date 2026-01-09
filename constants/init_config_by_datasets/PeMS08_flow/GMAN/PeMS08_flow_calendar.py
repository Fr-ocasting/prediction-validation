constant_name = ''
constant_modif = {}
SEED = 1

config = {'dataset_names' :['PeMS08_flow','calendar'],#['PeMS08_flow','calendar'], #['PeMS08_flow','PeMS08_speed','PeMS08_occupancy','calendar'],
          'target_data': 'PeMS08_flow',
          'dataset_for_coverage': ['PeMS08'],
          'calendar_types':['dayofweek', 'timeofday'],
          'model_name': 'GMAN',
          'use_target_as_context': False,
          'data_augmentation': False,
          'step_ahead': 12,
          'horizon_step': 1,
            'station' : [],
            'freq': '5min',
            'H':12,
            'D':0,
            'W':0,
            'train_prop': 0.6,
            'valid_prop': 0.2,
            'test_prop': 0.2,

            'standardize': True,
            'minmaxnorm': False,
            'unormalize_loss' : True,
            'loss_function_type': 'MSE',# 'HuberLoss',

              # GMAN parameters:
            'nb_STAttblocks': 1,
            'num_heads': 8,  # 'num_heads = K = 8' in the paper
            'head_dim': 8,   # 'head_dim = d = 8' in the paper
            'bn_decay': 0.1,
            'adj_type': 'dist',

            # Optimizer: 
            'batch_size': 32, # 16, 32, 64
            'epochs':100,
            'optimizer': 'adam',
            'lr': 0.001, # 0.001
            'weight_decay': 0.0015,
            'dropout': 0.0,

            'metrics':['masked_mae','masked_rmse','masked_mape','masked_mse','mae','rmse','mape','mse','mase'],
            }
