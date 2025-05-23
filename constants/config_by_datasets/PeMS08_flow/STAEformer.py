SEED = 1

config = {'dataset_names' :['PeMS08_flow','calendar'],#['PeMS08_flow','calendar'], #['PeMS08_flow','PeMS08_speed','PeMS08_occupancy','calendar'],
          'target_data': 'PeMS08_flow',
          'dataset_for_coverage': ['PeMS08'],
          'model_name': 'STAEformer',
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
            'epochs':300,
            'optimizer': 'adamw',
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
