SEED = 1

config = {'dataset_names' : ['subway_in','calendar','netmob_POIs'],
          'target_data': 'subway_in',
          'dataset_for_coverage': ['subway_in','netmob_POIs'],
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
            'optimizer': 'adamw', #adam
            'lr': 0.001,
            'weight_decay': 0.0015,
            'loss_function_type':'HuberLoss',

            # 'torch_scheduler_type': 'MultiStepLR',
            # 'torch_scheduler_milestone': [25, 45, 65],
            # 'torch_scheduler_gamma':  0.1,

            'torch_scheduler_milestone': 20,
            'torch_scheduler_gamma':0.9925,
            'torch_scheduler_type': 'warmup',
            'torch_scheduler_lr_start_factor': 0.3,

            'train_prop': 0.6,
            'valid_prop': 0.2,
            'test_prop': 0.2,
            'dropout':0.1,
            'standardize': True,
            'minmaxnorm': False,
            'metrics':['masked_mae','masked_rmse','masked_mape','masked_mse','mae','rmse','mape','mse','mase'],

            'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                                        'stacked_contextual': True,
                                        'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                                        'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                        'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                        'NetMob_expanded' : '', # '' # '_expanded'
                                        'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                        'vision_model_name' : None,
                                        'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                        'attn_kwargs': {'latent_dim' : 4 ,
                                                        'dim_feedforward' : 64,
                                                        'num_heads' : 3 ,
                                                        'dim_model' : 48,}  
                                        #'H' : ,
                                        #'D': ,
                                        #'W': , 
                            },
                        # 'netmob_POIs': {'need_global_attn':False, 
                        #                 'stacked_contextual': True,
                        #                 'NetMob_selected_apps' : ['Deezer','Google_Maps'],
                        #                   'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                   'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                   'NetMob_expanded' : '', # '' # '_expanded'
                        #                   'NetMob_only_epsilon': True, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                 'vision_model_name' : None
                        #                  },

                        'subway_out': {'need_global_attn':False, 
                                        'stacked_contextual': True,
                                        'vision_model_name': None, # Define the type of model used to extract contextual information from NetMob
                                        'vision_input_type': None, # 'image_per_stations' # 'unique_image_through_lyon'  
                                        'grn_out_dim': 0, # If >0 then stack a GRN layer to the output module
                                        'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                            },
                        }
            }
