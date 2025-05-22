SEED = 1

config = {'dataset_names' : ['subway_in','calendar'],
          'target_data': 'subway_in',
          'dataset_for_coverage': ['subway_in','netmob_image_per_station'],
          'model_name': 'STAEformer',
          'use_target_as_context': False,
          'data_augmentation': False,
          'step_ahead': 4,
            'station' : [],
            'freq': '15min',
            'H':6,
            'D':1,
            'W':0,
            'input_embedding_dim': 24,
            'tod_embedding_dim': 24,
            'dow_embedding_dim': 24,
            'spatial_embedding_dim':0,
            'adaptive_embedding_dim': 80,

            'node_dim': 64,
            'feed_forward_dim': 256,
            'num_heads': 4,
            'num_layers': 3,

            'use_mixed_proj': True,

            'calendar_types':['dayofweek', 'timeofday'],


            'batch_size': 16, # 16, 32, 64
            'epochs':3,
            'optimizer': 'adam',
            'lr': 0.001,
            'weight_decay': 0.0015,
            'torch_scheduler_type': 'MultiStepLR',
            'loss_function_type':'HuberLoss',
            'torch_scheduler_milestone': [25, 45, 65],
            'torch_scheduler_gamma':0.1,
            'train_prop': 0.6,
            'valid_prop': 0.2,
            'test_prop': 0.2,
            'dropout':0.1,
            'standardize': True,
            'minmaxnorm': False,
            'metrics':['masked_mae','masked_rmse','masked_mape','masked_mse','mae','rmse','mape','mse','mase'],
            }
