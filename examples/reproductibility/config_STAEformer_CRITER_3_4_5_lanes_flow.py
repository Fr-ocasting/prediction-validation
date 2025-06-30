# Modif CRITER - STAEformer

constant_name = ''
constant_modif = {'dataset_for_coverage': ['CRITER_3_4_5_lanes_flow','netmob_POIs'],
                  'target_data': 'CRITER_3_4_5_lanes_flow',
                  'model_name': 'STAEformer',
                        'optimizer': 'adamw',
                        'batch_size': 128,
                        'epochs': 500,
                        'step_ahead' : 10,
                        'horizon_step' : 5,
                        'freq': '6min',
                        'H':10,
                        'D':0,
                        'W':0,
                        }


modifications = { 'Init': {'dataset_names' :['CRITER_3_4_5_lanes_flow','calendar','netmob_POIs'], # ['CRITER_3_4_5_lanes_flow','calendar','netmob_POIs'], # ['CRITER_3_4_5_lanes_flow','calendar'],
                          'use_target_as_context': False,
                          'data_augmentation': False,
                          'stacked_contextual': False,
                            'station' : [],

                            'input_embedding_dim': 24,
                            'tod_embedding_dim': 0,
                            'dow_embedding_dim': 0,
                            'spatial_embedding_dim':0,
                            'adaptive_embedding_dim': 80,

                            'node_dim': 64,
                            'feed_forward_dim': 256,
                            'num_heads': 4,
                            'num_layers': 3,

                            'use_mixed_proj': True,

                            'calendar_types':['dayofweek', 'timeofday'],

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
                            
                            'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
                                                                                            'stacked_contextual': True,
                                                                                            'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
                                                                                              'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                              'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                              'NetMob_expanded' : '', # '' # '_expanded'
                                                                                              'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                              'epsilon_clustering': 0.15, 
                                                                                              'vision_model_name' : None,
                                                                                              'use_only_for_common_dates': True, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                              'attn_kwargs': {'latent_dim' : 4 ,
                                                                                                              'dim_feedforward' : 64,
                                                                                                              'num_heads' : 3 ,
                                                                                                              'dim_model' : 48,
                                                                                                              'keep_topk': False}  
                                                                                              #'H' : ,
                                                                                              #'D': ,
                                                                                              #'W': , 
                                                                                          }
                                                                                  },
                            'denoising_names':[]
                                },

              'calendar': {'dataset_names' :['CRITER_3_4_5_lanes_flow','calendar','netmob_POIs'], # ['CRITER_3_4_5_lanes_flow','calendar','netmob_POIs'], # ['CRITER_3_4_5_lanes_flow','calendar'],
                            'use_target_as_context': False,
                            'data_augmentation': False,
                            'stacked_contextual': False,
                              'station' : [],

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
                              
                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
                                                                                            'stacked_contextual': True,
                                                                                            'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
                                                                                              'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                              'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                              'NetMob_expanded' : '', # '' # '_expanded'
                                                                                              'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                              'epsilon_clustering': 0.15, 
                                                                                              'vision_model_name' : None,
                                                                                              'use_only_for_common_dates': True, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                              'attn_kwargs': {'latent_dim' : 4 ,
                                                                                                              'dim_feedforward' : 64,
                                                                                                              'num_heads' : 3 ,
                                                                                                              'dim_model' : 48,
                                                                                                              'keep_topk': False}  
                                                                                              #'H' : ,
                                                                                              #'D': ,
                                                                                              #'W': , 
                                                                                          }
                                                                                  },
                              'denoising_names':[],
                              },

'calendar_Google_Maps_Deezer_IRIS_clustering015': {'dataset_names' :['CRITER_3_4_5_lanes_flow','calendar','netmob_POIs'], # ['CRITER_3_4_5_lanes_flow','calendar','netmob_POIs'], # ['CRITER_3_4_5_lanes_flow','calendar'],
                                                    'use_target_as_context': False,
                                                    'data_augmentation': False,
                                                    'stacked_contextual': False,

                                                      'station' : [],

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
                                                      
                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
                                                                                            'stacked_contextual': True,
                                                                                            'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
                                                                                              'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                              'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                              'NetMob_expanded' : '', # '' # '_expanded'
                                                                                              'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                              'epsilon_clustering': 0.15, 
                                                                                              'vision_model_name' : None,
                                                                                              'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                              'attn_kwargs': {'latent_dim' : 4 ,
                                                                                                              'dim_feedforward' : 64,
                                                                                                              'num_heads' : 3 ,
                                                                                                              'dim_model' : 48,
                                                                                                              'keep_topk': False}  
                                                                                              #'H' : ,
                                                                                              #'D': ,
                                                                                              #'W': , 
                                                                                          }
                                                                                  },
                                                    'denoising_names':[]
                                                    },     




         }



if len(constant_modif) > 0:
    modifications_bis = modifications.copy()
    modifications = {}
    for key, value in modifications_bis.items():
        modif_i = constant_modif.copy()
        modif_i.update(value)
        modifications[f"{constant_name}_{key}"] = modif_i