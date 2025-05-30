SEED = 1

config = {'dataset_names' : ['CRITER_3_4_5_lanes_flow','calendar','netmob_POIs'],
          'target_data': 'CRITER_3_4_5_lanes_flow',
          'dataset_for_coverage': ['CRITER_3_4_5_lanes_flow','netmob_POIs'],
          'model_name': 'STAEformer',
          'use_target_as_context': False,
          'data_augmentation': False,
          'step_ahead': 10,
            'station' : [],
            'freq': '6min',
            'H':10,
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


            'batch_size': 128, # 16, 32, 64
            'epochs':300,
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

            'NetMob_only_epsilon': True,    # True # False
            'NetMob_selected_apps': ['Deezer','Google_Maps'],# ['Apple_iMessage','Web_Ads'], #,'Web_Weather','Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
            'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
            'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
            'NetMob_expanded' : '', # '' # '_expanded
            }
