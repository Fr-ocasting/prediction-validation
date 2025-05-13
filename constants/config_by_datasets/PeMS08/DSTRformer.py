SEED = 1

config = {'dataset_for_coverage' : ['PeMS08'],
          'target_data': 'PeMS08',
          'dataset_for_coverage': ['PeMS08'],
          'model_name': 'DSTRformer',
          'use_target_as_context': False,
          'data_augmentation': False,
          'step_ahead': 12,
        
            'station' : [],
            'freq': '5min',
            'H':12,
            'D':0,
            'W':0,
            'input_embedding_dim': 24,
            'tod_embedding_dim': 24,
            'ts_embedding_dim': 28,
            'dow_embedding_dim': 24,
            'time_embedding_dim': 0,
            'adaptive_embedding_dim': 100,
            'node_dim': 64,
            'feed_forward_dim': 256,
            'out_feed_forward_dim': 256,
            'mlp_num_layers': 2,
            'num_heads': 4,
            'num_layers': 2,
            'num_layers_m': 1,
            'dropout': 0.1,
            'use_mixed_proj': True,
            "adj_type": 'adj', # choices = ['adj','dist','corr']
            "adj_normalize_method": 'doubletransition', # choices = ['normlap','scalap','symadj','transition','doubletransition','identity']
            "threshold": None, # choices = [0.5, 0.7, 0.9] # useless if adj_type = 'adj'
            'calendar_types':['dayofweek', 'timeofday'],


            'batch_size': 16, # 16, 32, 64
            'epochs':80,
            'optimizer': 'adam',
            'lr': 0.001,
            'weight_decay': 0.0015,
            'torch_scheduler_type': 'MultiStepLR',
            'loss_function_type':'masked_mae',
            'torch_scheduler_milestone': [25, 45, 65],
            'torch_scheduler_gamma':0.1,
            'train_prop': 0.6,
            'valid_prop': 0.2,
            'test_prop': 0.2,
            'standardize': True,
            'minmaxnorm': False,
            'metrics':['masked_mae','masked_rmse','masked_mape','masked_mse','mae','rmse','mape','mse','mase'],
            }


