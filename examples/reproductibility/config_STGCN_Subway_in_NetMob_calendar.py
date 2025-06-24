# Modif Subway-in-NetMob-calendar  STAEformer

                        # 'torch_scheduler_milestone': 20,
                        # 'torch_scheduler_gamma':0.9925,
                        # 'torch_scheduler_type': 'warmup',
                        # 'torch_scheduler_lr_start_factor': 0.3,


# Si netmob autour de epsilon m : 
# NetMob_selected_tags isin ['station_epsilon100','station_epsilon300']
# 'NetMob_only_epsilon' =  True

# Sinon : 
# NetMob_selected_tags isin ['iris','station','stadium','university','park','shop','nightclub','parkings','theatre','transit','public_transport']
# 'NetMob_only_epsilon' =  False

constant_name = ''
constant_modif = {'dataset_for_coverage': ['subway_in','netmob_POIs'],
                  'target_data': 'subway_in',
                        'optimizer': 'adamw',
                        'batch_size': 128,
                        'epochs':300,

                        'freq': '15min',
                        'H':6,
                        'D':1,
                        'W':0,
                        }

modifications = {




######==========================================================================
#        BEST MODEL OBTENU 
#######========================================================================

# # All Steps RMSE = 38.38, MAE = 21.57, MAPE = 25.75, MSE = 1474.81
#  'subway_in_calendar_emb64_out64_Huber_MinMax': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs':500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                    'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,
#                                                     },




# # All Steps RMSE = 38.78, MAE = 21.95, MAPE = 27.27, MSE = 1506.55
#  'subway_in_calendar_emb64_out64_Huber_Standardize': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs':500,
#                                                     'standardize': True,
#                                                     'minmaxnorm': False,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                    'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,
#                                                     },


# # [Deezer], MinMax
# # All Steps RMSE = 39.01, MAE = 22.00, MAPE = 25.67, MSE = 1525.65
# # All Steps RMSE = 39.23, MAE = 21.93, MAPE = 25.78, MSE = 1540.95
# # All Steps RMSE = 39.23, MAE = 22.11, MAPE = 25.63, MSE = 1541.52
'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_01': {'target_data': 'subway_in',
                                                  'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
                                                  'embedding_calendar_types': ['dayofweek', 'hour'],
                                                  'loss_function_type':'HuberLoss',
                                                   'Kt': 2,
                                                    'stblock_num': 4,
                                                    'Ks': 2,
                                                    'graph_conv_type': 'graph_conv',
                                                    'gso_type': 'sym_renorm_adj',
                                                    'enable_bias': True,
                                                    'adj_type': 'corr',
                                                    'enable_padding': True,
                                                    'threshold': 0.3,
                                                    'act_func': 'glu',
                                                    'temporal_h_dim': 64,
                                                    'spatial_h_dim': 256,
                                                    'output_h_dim': 64,
                                                    'weight_decay': 0.0014517707449388,
                                                    'batch_size': 128,
                                                    'lr': 0.00071,
                                                    'dropout': 0.145169206052754,
                                                    'epochs': 500,
                                                    'standardize': False,
                                                    'minmaxnorm': True,

                                                    'TE_embedding_dim': 64,
                                                    'TE_out_h_dim': 64,
                                                    'TE_concatenation_late': True,
                                                    'TE_concatenation_early':False,

                                                     'optimizer': 'adamw',
                                                    'batch_size': 128,
                                                    'freq': '15min',
                                                    'H':6,
                                                    'D':1,
                                                    'W':0,

                                                     'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
                                                                                            'stacked_contextual': True,
                                                                                            'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                                                                                            'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                            'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                            'NetMob_expanded' : '', # '' # '_expanded'
                                                                                            'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                            'epsilon_clustering': 0.1, 
                                                                                            'vision_model_name' : None,
                                                                                            'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                            'attn_kwargs': {'latent_dim': 2 ,
                                                                                                            'dim_feedforward' : 64,
                                                                                                            'num_heads':  2 ,
                                                                                                            'dim_model':  64,
                                                                                                            'keep_topk': False}  
                                                                                            #'H' : ,
                                                                                            #'D': ,
                                                                                            #'W': , 
                                                                                },
                                                                            },  
                                                    'denoising_names':['netmob_POIs'],
                                                    'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                                    'denoising_modes':["train","valid","test"],             # par défaut
                                                    'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                    },

# # [Deezer], MinMax, Netmob: STD
# All Steps RMSE = 40.40, MAE = 22.61, MAPE = 27.11, MSE = 1637.98
# All Steps RMSE = 39.64, MAE = 22.26, MAPE = 27.29, MSE = 1572.82
# All Steps RMSE = 40.41, MAE = 22.30, MAPE = 25.34, MSE = 1635.67
'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_01_std_netmob': {'target_data': 'subway_in',
                                                  'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
                                                  'embedding_calendar_types': ['dayofweek', 'hour'],
                                                  'loss_function_type':'HuberLoss',
                                                   'Kt': 2,
                                                    'stblock_num': 4,
                                                    'Ks': 2,
                                                    'graph_conv_type': 'graph_conv',
                                                    'gso_type': 'sym_renorm_adj',
                                                    'enable_bias': True,
                                                    'adj_type': 'corr',
                                                    'enable_padding': True,
                                                    'threshold': 0.3,
                                                    'act_func': 'glu',
                                                    'temporal_h_dim': 64,
                                                    'spatial_h_dim': 256,
                                                    'output_h_dim': 64,
                                                    'weight_decay': 0.0014517707449388,
                                                    'batch_size': 128,
                                                    'lr': 0.00071,
                                                    'dropout': 0.145169206052754,
                                                    'epochs': 500,
                                                    'standardize': False,
                                                    'minmaxnorm': True,

                                                    'TE_embedding_dim': 64,
                                                    'TE_out_h_dim': 64,
                                                    'TE_concatenation_late': True,
                                                    'TE_concatenation_early':False,

                                                     'optimizer': 'adamw',
                                                    'batch_size': 128,
                                                    'freq': '15min',
                                                    'H':6,
                                                    'D':1,
                                                    'W':0,

                                                     'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
                                                                                            'stacked_contextual': True,
                                                                                            'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                                                                                            'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                            'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                            'NetMob_expanded' : '', # '' # '_expanded'
                                                                                            'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                            'epsilon_clustering': 0.1, 
                                                                                            'vision_model_name' : None,
                                                                                            'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                            'minmaxnorm': False,
                                                                                            'standardize': True,
                                                                                            'attn_kwargs': {'latent_dim': 2 ,
                                                                                                            'dim_feedforward' : 64,
                                                                                                            'num_heads':  2 ,
                                                                                                            'dim_model':  64,
                                                                                                            'keep_topk': False}  
                                                                                            #'H' : ,
                                                                                            #'D': ,
                                                                                            #'W': , 
                                                                                },
                                                                            },  
                                                    'denoising_names':['netmob_POIs'],
                                                    'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                                    'denoising_modes':["train","valid","test"],             # par défaut
                                                    'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                    },
# # [Deezer], MinMax, Netmob: MinMax
# All Steps RMSE = 39.37, MAE = 22.18, MAPE = 26.07, MSE = 1553.27
# All Steps RMSE = 39.33, MAE = 22.07, MAPE = 25.45, MSE = 1550.40
# All Steps RMSE = 39.31, MAE = 22.23, MAPE = 27.01, MSE = 1548.29
'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_01_minmax_netmob': {'target_data': 'subway_in',
                                                  'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
                                                  'embedding_calendar_types': ['dayofweek', 'hour'],
                                                  'loss_function_type':'HuberLoss',
                                                   'Kt': 2,
                                                    'stblock_num': 4,
                                                    'Ks': 2,
                                                    'graph_conv_type': 'graph_conv',
                                                    'gso_type': 'sym_renorm_adj',
                                                    'enable_bias': True,
                                                    'adj_type': 'corr',
                                                    'enable_padding': True,
                                                    'threshold': 0.3,
                                                    'act_func': 'glu',
                                                    'temporal_h_dim': 64,
                                                    'spatial_h_dim': 256,
                                                    'output_h_dim': 64,
                                                    'weight_decay': 0.0014517707449388,
                                                    'batch_size': 128,
                                                    'lr': 0.00071,
                                                    'dropout': 0.145169206052754,
                                                    'epochs': 500,
                                                    'standardize': False,
                                                    'minmaxnorm': True,

                                                    'TE_embedding_dim': 64,
                                                    'TE_out_h_dim': 64,
                                                    'TE_concatenation_late': True,
                                                    'TE_concatenation_early':False,

                                                     'optimizer': 'adamw',
                                                    'batch_size': 128,
                                                    'freq': '15min',
                                                    'H':6,
                                                    'D':1,
                                                    'W':0,

                                                     'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
                                                                                            'stacked_contextual': True,
                                                                                            'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                                                                                            'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                            'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                            'NetMob_expanded' : '', # '' # '_expanded'
                                                                                            'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                            'epsilon_clustering': 0.1, 
                                                                                            'vision_model_name' : None,
                                                                                            'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                            'minmaxnorm': True,
                                                                                            'standardize': False,
                                                                                            'attn_kwargs': {'latent_dim': 2 ,
                                                                                                            'dim_feedforward' : 64,
                                                                                                            'num_heads':  2 ,
                                                                                                            'dim_model':  64,
                                                                                                            'keep_topk': False}  
                                                                                            #'H' : ,
                                                                                            #'D': ,
                                                                                            #'W': , 
                                                                                },
                                                                            },  
                                                    'denoising_names':['netmob_POIs'],
                                                    'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                                    'denoising_modes':["train","valid","test"],             # par défaut
                                                    'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                    },


# [Google Maps, Deezer], MinMax
# # All Steps RMSE = 38.96, MAE = 21.98, MAPE = 26.38, MSE = 1520.98
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_02_MinMax': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.2, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     }, 


# # [Deezer], MinMax
# # All Steps RMSE = 39.01, MAE = 22.00, MAPE = 25.67, MSE = 1525.65
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_01_MinMax': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.1, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

# # [Google Maps], MinMax 
# # All Steps RMSE = 39.00, MAE = 21.79, MAPE = 25.17, MSE = 1524.94
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_01_MinMax': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.1, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

# 



# #  [Google Maps, Deezer], Standardize
# # All Steps RMSE = 39.07, MAE = 22.07, MAPE = 26.21, MSE = 1527.94
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_01_standardize': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': True,
#                                                     'minmaxnorm': False,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.1, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

######==========================================================================
#        BEST MODEL OBTENU 
#######========================================================================



# # All Steps RMSE = 39.02, MAE = 21.90, MAPE = 24.27, MSE = 1525.39
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_015_MinMax': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

# # All Steps RMSE = 39.20, MAE = 22.02, MAPE = 25.80, MSE = 1539.31
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_02_MinMax': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.2, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },



# # All Steps RMSE = 39.60, MAE = 22.16, MAPE = 25.48, MSE = 1571.99
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_keep_topk': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': True}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },


# # All Steps RMSE = 39.39, MAE = 22.05, MAPE = 24.90, MSE = 1554.24
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_015_MinMax': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

# # All Steps RMSE = 39.37, MAE = 21.97, MAPE = 24.75, MSE = 1553.33
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_02_MinMax': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.2, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },       



# # All Steps RMSE = 39.12, MAE = 22.01, MAPE = 25.86, MSE = 1533.05
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_01_MinMax': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.1, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

# # All Steps RMSE = 40.02, MAE = 22.11, MAPE = 25.26, MSE = 1607.58
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_015_MinMax': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },


# # All Steps RMSE = 39.69, MAE = 22.13, MAPE = 26.56, MSE = 1580.04
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_015_standardize': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': True,
#                                                     'minmaxnorm': False,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

# # All Steps RMSE = 39.99, MAE = 22.21, MAPE = 25.52, MSE = 1603.00
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_02_standardize': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': True,
#                                                     'minmaxnorm': False,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.2, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     }, 


# # All Steps RMSE = 40.30, MAE = 22.39, MAPE = 26.15, MSE = 1627.60
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_01_standardize': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': True,
#                                                     'minmaxnorm': False,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.1, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

# # All Steps RMSE = 39.84, MAE = 22.14, MAPE = 25.67, MSE = 1590.85
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_015_standardize': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': True,
#                                                     'minmaxnorm': False,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

# # All Steps RMSE = 39.99, MAE = 22.19, MAPE = 26.14, MSE = 1603.52
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_02_standardize': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': True,
#                                                     'minmaxnorm': False,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.2, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },


# # All Steps RMSE = 39.71, MAE = 22.26, MAPE = 25.46, MSE = 1580.75
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_01_standardize': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': True,
#                                                     'minmaxnorm': False,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.1, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

# # All Steps RMSE = 39.89, MAE = 22.35, MAPE = 25.64, MSE = 1594.36
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_015_standardize': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': True,
#                                                     'minmaxnorm': False,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

# # All Steps RMSE = 40.83, MAE = 22.99, MAPE = 27.67, MSE = 1670.92
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_02_standardize': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': True,
#                                                     'minmaxnorm': False,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.2, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },       




# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim48_ff128_h3_l4_exp_smooth08_keep_topk': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 4 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': True}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim64_ff128_h4_l4_exp_smooth08_keep_topk': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 4 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  4 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': True}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim128_ff128_h4_l4_exp_smooth08_keep_topk': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 4 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  4 ,
#                                                                                                             'dim_model':  128,
#                                                                                                             'keep_topk': True}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim64_ff128_h4_l4_exp_smooth08': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 4 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  4 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim128_ff128_h4_l4_exp_smooth08': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 4 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  4 ,
#                                                                                                             'dim_model':  128,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim64_ff128_h4_l4_exp_smooth08_clustering01': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.1, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 4 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  4 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim128_ff128_h4_l4_exp_smooth08_clustering01': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.1, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 4 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  4 ,
#                                                                                                             'dim_model':  128,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

#    'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_keep_topk_clustering01': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.1, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': True}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },



# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering01': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.1, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_keep_topk_clustering01': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.1, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': True}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },     


# 'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_Instagram_Uber_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_keep_topk_clustering01': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Deezer','Google_Maps','Uber','Instagram'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.1, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': True}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':['netmob_POIs'],
#                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                     'denoising_modes':["train","valid","test"],             # par défaut
#                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                                                     },

#  'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_keep_topk_clustering01': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.1, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': True}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     },

#  'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_keep_topk_clustering015': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
#                                                   'embedding_calendar_types': ['dayofweek', 'hour'],
#                                                   'loss_function_type':'HuberLoss',
#                                                    'Kt': 2,
#                                                     'stblock_num': 4,
#                                                     'Ks': 2,
#                                                     'graph_conv_type': 'graph_conv',
#                                                     'gso_type': 'sym_renorm_adj',
#                                                     'enable_bias': True,
#                                                     'adj_type': 'corr',
#                                                     'enable_padding': True,
#                                                     'threshold': 0.3,
#                                                     'act_func': 'glu',
#                                                     'temporal_h_dim': 64,
#                                                     'spatial_h_dim': 256,
#                                                     'output_h_dim': 64,
#                                                     'weight_decay': 0.0014517707449388,
#                                                     'batch_size': 128,
#                                                     'lr': 0.00071,
#                                                     'dropout': 0.145169206052754,
#                                                     'epochs': 500,
#                                                     'standardize': False,
#                                                     'minmaxnorm': True,

#                                                     'TE_embedding_dim': 64,
#                                                     'TE_out_h_dim': 64,
#                                                     'TE_concatenation_late': True,
#                                                     'TE_concatenation_early':False,

#                                                      'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2 ,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': True}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     },








    # 'subway_in_MSE_MinMax': {'target_data': 'subway_in',
    #                                       'dataset_names': ['subway_in'],
    #                                 'loss_function_type':'MSE',
    #                                 'Kt': 2,
    #                                 'stblock_num': 2,
    #                                 'Ks': 2,
    #                                 'graph_conv_type': 'graph_conv',
    #                                 'gso_type': 'sym_renorm_adj',
    #                                 'enable_bias': True,
    #                                 'adj_type': 'corr',
    #                                 'enable_padding': True,
    #                                 'threshold': 0.3,
    #                                 'act_func': 'glu',
    #                                 'temporal_h_dim': 256,
    #                                 'spatial_h_dim': 64,
    #                                 'output_h_dim': 32,

    #                                 'weight_decay': 0.0780888601156736,
    #                                 'batch_size': 128,
    #                                 'lr': 0.00084,
    #                                 'dropout': 0.1702481119406736,
    #                                 'epochs': 500,

    #                                 'standardize': False,
    #                                 'minmaxnorm': True,
    #                                 },
    #             'subway_in_calendar_MSE_MinMax': {'target_data': 'subway_in',
    #                                               'dataset_names': ['subway_in', 'calendar_embedding'],
    #                                               'embedding_calendar_types': ['dayofweek', 'hour'],
    #                                               'loss_function_type':'MSE',
    #                                                'Kt': 2,
    #                                                 'stblock_num': 4,
    #                                                 'Ks': 2,
    #                                                 'graph_conv_type': 'graph_conv',
    #                                                 'gso_type': 'sym_renorm_adj',
    #                                                 'enable_bias': True,
    #                                                 'adj_type': 'corr',
    #                                                 'enable_padding': True,
    #                                                 'threshold': 0.3,
    #                                                 'act_func': 'glu',
    #                                                 'temporal_h_dim': 64,
    #                                                 'spatial_h_dim': 256,
    #                                                 'output_h_dim': 64,
    #                                                 'weight_decay': 0.0014517707449388,
    #                                                 'batch_size': 128,
    #                                                 'lr': 0.00071,
    #                                                 'dropout': 0.145169206052754,
    #                                                 'epochs': 500,
    #                                                 'standardize': False,
    #                                                 'minmaxnorm': True,
    #                                                 },
                                                    
            # 'subway_in_Huber_MinMax_config_2': {'target_data': 'subway_in',
            #                               'dataset_names': ['subway_in'],
            #                         'loss_function_type':'HuberLoss',
            #                         'Kt': 2,
            #                         'stblock_num': 2,
            #                         'Ks': 2,
            #                         'graph_conv_type': 'graph_conv',
            #                         'gso_type': 'sym_renorm_adj',
            #                         'enable_bias': True,
            #                         'adj_type': 'corr',
            #                         'enable_padding': True,
            #                         'threshold': 0.3,
            #                         'act_func': 'glu',
            #                         'temporal_h_dim': 256,
            #                         'spatial_h_dim': 64,
            #                         'output_h_dim': 32,

            #                         'weight_decay': 0.0780888601156736,
            #                         'batch_size': 128,
            #                         'lr': 0.00084,
            #                         'dropout': 0.1702481119406736,
            #                         'epochs': 500,

            #                         'standardize': False,
            #                         'minmaxnorm': True,
            #                         },
            # 'subway_in_calendar_emb4_out8_Huber_MinMax_config_2': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                         'loss_function_type':'HuberLoss',
            #                         'Kt': 2,
            #                         'stblock_num': 2,
            #                         'Ks': 2,
            #                         'graph_conv_type': 'graph_conv',
            #                         'gso_type': 'sym_renorm_adj',
            #                         'enable_bias': True,
            #                         'adj_type': 'corr',
            #                         'enable_padding': True,
            #                         'threshold': 0.3,
            #                         'act_func': 'glu',
            #                         'temporal_h_dim': 256,
            #                         'spatial_h_dim': 64,
            #                         'output_h_dim': 32,

            #                         'weight_decay': 0.0780888601156736,
            #                         'batch_size': 128,
            #                         'lr': 0.00084,
            #                         'dropout': 0.1702481119406736,
            #                         'epochs': 500,

            #                         'standardize': False,
            #                         'minmaxnorm': True,

            #                         'TE_embedding_dim': 4,
            #                         'TE_out_h_dim': 8,
            #                         'TE_concatenation_late': True,
            #                         'TE_concatenation_early':False,
            #                         },

            #     'subway_in_calendar_emb4_out8_Huber_MinMax': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': False,
            #                                         'minmaxnorm': True,

            #                                         'TE_embedding_dim': 4,
            #                                         'TE_out_h_dim': 8,
            #                                         'TE_concatenation_late': True,
            #                                         'TE_concatenation_early':False,
            #                                         },

            # 'subway_in_calendar_emb8_out8_Huber_MinMax': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': False,
            #                                         'minmaxnorm': True,

            #                                         'TE_embedding_dim': 8,
            #                                         'TE_out_h_dim': 8,
            #                                         'TE_concatenation_late': True,
            #                                         'TE_concatenation_early':False,
            #                                         },

            # 'subway_in_calendar_emb8_out16_Huber_MinMax': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': False,
            #                                         'minmaxnorm': True,

            #                                         'TE_embedding_dim': 8,
            #                                         'TE_out_h_dim': 16,
            #                                         'TE_concatenation_late': True,
            #                                         'TE_concatenation_early':False,
            #                                         },

            # 'subway_in_calendar_emb16_out16_Huber_MinMax': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': False,
            #                                         'minmaxnorm': True,

            #                                         'TE_embedding_dim': 16,
            #                                         'TE_out_h_dim': 16,
            #                                         'TE_concatenation_late': True,
            #                                         'TE_concatenation_early':False,
            #                                         },
            # 'subway_in_calendar_emb16_out32_Huber_MinMax': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': False,
            #                                         'minmaxnorm': True,

            #                                         'TE_embedding_dim': 16,
            #                                         'TE_out_h_dim': 32,
            #                                         'TE_concatenation_late': True,
            #                                         'TE_concatenation_early':False,
            #                                         },

            #     'subway_in_calendar_emb64_out32_Huber_MinMax': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': False,
            #                                         'minmaxnorm': True,

            #                                         'TE_embedding_dim': 64,
            #                                         'TE_out_h_dim': 32,
            #                                         'TE_concatenation_late': True,
            #                                         'TE_concatenation_early':False,
            #                                         },
            #     'subway_in_calendar_emb32_out16_Huber_MinMax': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': False,
            #                                         'minmaxnorm': True,

            #                                         'TE_embedding_dim': 32,
            #                                         'TE_out_h_dim': 16,
            #                                         'TE_concatenation_late': True,
            #                                         'TE_concatenation_early':False,
            #                                         },
            #     'subway_in_calendar_emb32_out32_Huber_MinMax': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': False,
            #                                         'minmaxnorm': True,

            #                                         'TE_embedding_dim': 32,
            #                                         'TE_out_h_dim': 32,
            #                                         'TE_concatenation_late': True,
            #                                         'TE_concatenation_early':False,
            #                                         },
            #     'subway_in_calendar_emb64_out64_Huber_MinMax': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': False,
            #                                         'minmaxnorm': True,

            #                                         'TE_embedding_dim': 64,
            #                                         'TE_out_h_dim': 64,
            #                                         'TE_concatenation_late': True,
            #                                         'TE_concatenation_early':False,
            #                                         },
                # 'subway_in_calendar_emb64_out64_Huber_Standardize_bis': {'target_data': 'subway_in',
                #                                   'dataset_names': ['subway_in', 'calendar_embedding'],
                #                                   'embedding_calendar_types': ['dayofweek', 'hour'],
                #                                   'loss_function_type':'HuberLoss',
                #                                    'Kt': 2,
                #                                     'stblock_num': 4,
                #                                     'Ks': 2,
                #                                     'graph_conv_type': 'graph_conv',
                #                                     'gso_type': 'sym_renorm_adj',
                #                                     'enable_bias': True,
                #                                     'adj_type': 'corr',
                #                                     'enable_padding': True,
                #                                     'threshold': 0.3,
                #                                     'act_func': 'glu',
                #                                     'temporal_h_dim': 64,
                #                                     'spatial_h_dim': 256,
                #                                     'output_h_dim': 64,
                #                                     'weight_decay': 0.0014517707449388,
                #                                     'batch_size': 128,
                #                                     'lr': 0.00071,
                #                                     'dropout': 0.145169206052754,
                #                                     'epochs': 500,
                #                                     'standardize': True,
                #                                     'minmaxnorm': False,

                #                                     'TE_embedding_dim': 64,
                #                                     'TE_out_h_dim': 64,
                #                                     'TE_concatenation_late': True,
                #                                     'TE_concatenation_early':False,
                #                                     },

                # 'subway_in_calendar_emb64_out64_Huber_MinMax_bis': {'target_data': 'subway_in',
                #                                   'dataset_names': ['subway_in', 'calendar_embedding'],
                #                                   'embedding_calendar_types': ['dayofweek', 'hour'],
                #                                   'loss_function_type':'HuberLoss',
                #                                    'Kt': 2,
                #                                     'stblock_num': 4,
                #                                     'Ks': 2,
                #                                     'graph_conv_type': 'graph_conv',
                #                                     'gso_type': 'sym_renorm_adj',
                #                                     'enable_bias': True,
                #                                     'adj_type': 'corr',
                #                                     'enable_padding': True,
                #                                     'threshold': 0.3,
                #                                     'act_func': 'glu',
                #                                     'temporal_h_dim': 64,
                #                                     'spatial_h_dim': 256,
                #                                     'output_h_dim': 64,
                #                                     'weight_decay': 0.0014517707449388,
                #                                     'batch_size': 128,
                #                                     'lr': 0.00071,
                #                                     'dropout': 0.145169206052754,
                #                                     'epochs': 500,
                #                                     'standardize': False,
                #                                     'minmaxnorm': True,

                #                                     'TE_embedding_dim': 64,
                #                                     'TE_out_h_dim': 64,
                #                                     'TE_concatenation_late': True,
                #                                     'TE_concatenation_early':False,
                #                                     },

        #         'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim48_ff64_h3_l4_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim' : 4 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads' : 3 ,
        #                                                                                                     'dim_model' : 48,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim48_ff64_h3_l4_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim' : 4 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads' : 3 ,
        #                                                                                                     'dim_model' : 48,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_Google_Maps_IRIS_attn_dim48_ff64_h3_l4_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim' : 4 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads' : 3 ,
        #                                                                                                     'dim_model' : 48,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
        #                                             },


        #         'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim48_ff64_h3_l4': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim' : 4 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads' : 3 ,
        #                                                                                                     'dim_model' : 48,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim48_ff64_h3_l4': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim' : 4 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads' : 3 ,
        #                                                                                                     'dim_model' : 48,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_Google_Maps_IRIS_attn_dim48_ff64_h3_l4': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim' : 4 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads' : 3 ,
        #                                                                                                     'dim_model' : 48,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },


        #         'subway_in_calendar_emb64_out64_Huber_Standardize_Deezer_IRIS_attn_dim48_ff64_h3_l4': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim' : 4 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads' : 3 ,
        #                                                                                                     'dim_model' : 48,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_Standardize_Google_Maps_IRIS_attn_dim48_ff64_h3_l4': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim' : 4 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads' : 3 ,
        #                                                                                                     'dim_model' : 48,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_Standardize_Deezer_Google_Maps_IRIS_attn_dim48_ff64_h3_l4': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim' : 4 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads' : 3 ,
        #                                                                                                     'dim_model' : 48,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },


        #         'subway_in_calendar_emb64_out64_Huber_Standardize_Deezer_IRIS_attn_dim48_ff64_h3_l4_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim' : 4 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads' : 3 ,
        #                                                                                                     'dim_model' : 48,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}     
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_Standardize_Google_Maps_IRIS_attn_dim48_ff64_h3_l4_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim' : 4 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads' : 3 ,
        #                                                                                                     'dim_model' : 48,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}    
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_Standardize_Deezer_Google_Maps_IRIS_attn_dim48_ff64_h3_l4_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim' : 4 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads' : 3 ,
        #                                                                                                     'dim_model' : 48,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
        #                                             },        
































        #         'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim': 2 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads':  2 ,
        #                                                                                                     'dim_model':  64,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim': 2 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads':  2 ,
        #                                                                                                     'dim_model':  64,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim': 2 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads':  2 ,
        #                                                                                                     'dim_model':  64,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
        #                                             },


        #         'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim64_ff64_h2_l2': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim': 2 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads':  2 ,
        #                                                                                                     'dim_model':  64,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim64_ff64_h2_l2': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim': 2 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads':  2 ,
        #                                                                                                     'dim_model':  64,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_Google_Maps_IRIS_attn_dim64_ff64_h2_l2': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim': 2 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads':  2 ,
        #                                                                                                     'dim_model':  64,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },


        #         'subway_in_calendar_emb64_out64_Huber_Standardize_Deezer_IRIS_attn_dim64_ff64_h2_l2': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim': 2 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads':  2 ,
        #                                                                                                     'dim_model':  64,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_Standardize_Google_Maps_IRIS_attn_dim64_ff64_h2_l2': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim': 2 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads':  2 ,
        #                                                                                                     'dim_model':  64,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_Standardize_Deezer_Google_Maps_IRIS_attn_dim64_ff64_h2_l2': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim': 2 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads':  2 ,
        #                                                                                                     'dim_model':  64,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },


        #         'subway_in_calendar_emb64_out64_Huber_Standardize_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim': 2 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads':  2 ,
        #                                                                                                     'dim_model':  64,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}     
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_Standardize_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim': 2 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads':  2 ,
        #                                                                                                     'dim_model':  64,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}    
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_Standardize_Deezer_Google_Maps_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim': 2 ,
        #                                                                                                     'dim_feedforward' : 64,
        #                                                                                                     'num_heads':  2 ,
        #                                                                                                     'dim_model':  64,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
        #                                             },  






















        #         'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim128_ff128_h4_l4_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim':  4 ,
        #                                                                                                     'dim_feedforward':  128,
        #                                                                                                     'num_heads': 4 ,
        #                                                                                                     'dim_model':  128,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim128_ff128_h4_l4_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim':  4 ,
        #                                                                                                     'dim_feedforward':  128,
        #                                                                                                     'num_heads': 4 ,
        #                                                                                                     'dim_model':  128,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_Google_Maps_IRIS_attn_dim128_ff128_h4_l4_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim':  4 ,
        #                                                                                                     'dim_feedforward':  128,
        #                                                                                                     'num_heads': 4 ,
        #                                                                                                     'dim_model':  128,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
        #                                             },


        #         'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim128_ff128_h4_l4': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim':  4 ,
        #                                                                                                     'dim_feedforward':  128,
        #                                                                                                     'num_heads': 4 ,
        #                                                                                                     'dim_model':  128,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_IRIS_attn_dim128_ff128_h4_l4': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim':  4 ,
        #                                                                                                     'dim_feedforward':  128,
        #                                                                                                     'num_heads': 4 ,
        #                                                                                                     'dim_model':  128,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_Google_Maps_IRIS_attn_dim128_ff128_h4_l4': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': False,
        #                                             'minmaxnorm': True,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim':  4 ,
        #                                                                                                     'dim_feedforward':  128,
        #                                                                                                     'num_heads': 4 ,
        #                                                                                                     'dim_model':  128,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },


        #         'subway_in_calendar_emb64_out64_Huber_Standardize_Deezer_IRIS_attn_dim128_ff128_h4_l4': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim':  4 ,
        #                                                                                                     'dim_feedforward':  128,
        #                                                                                                     'num_heads': 4 ,
        #                                                                                                     'dim_model':  128,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_Standardize_Google_Maps_IRIS_attn_dim128_ff128_h4_l4': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim':  4 ,
        #                                                                                                     'dim_feedforward':  128,
        #                                                                                                     'num_heads': 4 ,
        #                                                                                                     'dim_model':  128,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_Standardize_Deezer_Google_Maps_IRIS_attn_dim128_ff128_h4_l4': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim':  4 ,
        #                                                                                                     'dim_feedforward':  128,
        #                                                                                                     'num_heads': 4 ,
        #                                                                                                     'dim_model':  128,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             },


        #         'subway_in_calendar_emb64_out64_Huber_Standardize_Deezer_IRIS_attn_dim128_ff128_h4_l4_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim':  4 ,
        #                                                                                                     'dim_feedforward':  128,
        #                                                                                                     'num_heads': 4 ,
        #                                                                                                     'dim_model':  128,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}     
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_Standardize_Google_Maps_IRIS_attn_dim128_ff128_h4_l4_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim':  4 ,
        #                                                                                                     'dim_feedforward':  128,
        #                                                                                                     'num_heads': 4 ,
        #                                                                                                     'dim_model':  128,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}    
        #                                             },

        #    'subway_in_calendar_emb64_out64_Huber_Standardize_Deezer_Google_Maps_IRIS_attn_dim128_ff128_h4_l4_exp_smooth08': {'target_data': 'subway_in',
        #                                           'dataset_names': ['subway_in', 'calendar_embedding'],
        #                                           'embedding_calendar_types': ['dayofweek', 'hour'],
        #                                           'loss_function_type':'HuberLoss',
        #                                            'Kt': 2,
        #                                             'stblock_num': 4,
        #                                             'Ks': 2,
        #                                             'graph_conv_type': 'graph_conv',
        #                                             'gso_type': 'sym_renorm_adj',
        #                                             'enable_bias': True,
        #                                             'adj_type': 'corr',
        #                                             'enable_padding': True,
        #                                             'threshold': 0.3,
        #                                             'act_func': 'glu',
        #                                             'temporal_h_dim': 64,
        #                                             'spatial_h_dim': 256,
        #                                             'output_h_dim': 64,
        #                                             'weight_decay': 0.0014517707449388,
        #                                             'batch_size': 128,
        #                                             'lr': 0.00071,
        #                                             'dropout': 0.145169206052754,
        #                                             'epochs': 500,
        #                                             'standardize': True,
        #                                             'minmaxnorm': False,

        #                                             'TE_embedding_dim': 64,
        #                                             'TE_out_h_dim': 64,
        #                                             'TE_concatenation_late': True,
        #                                             'TE_concatenation_early':False,

        #                                              'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
        #                                                                                     'stacked_contextual': True,
        #                                                                                     'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
        #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                                     'vision_model_name' : None,
        #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                                     'attn_kwargs': {'latent_dim':  4 ,
        #                                                                                                     'dim_feedforward':  128,
        #                                                                                                     'num_heads': 4 ,
        #                                                                                                     'dim_model':  128,}  
        #                                                                                     #'H' : ,
        #                                                                                     #'D': ,
        #                                                                                     #'W': , 
        #                                                                         },
        #                                                                     },  
        #                                             'denoising_names':['netmob_POIs'],
        #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                             'denoising_modes':["train","valid","test"],             # par défaut
        #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
        #                                             },  

            #     'subway_in_calendar_Huber_standardize': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,
            #                                         },
            #     'subway_in_calendar_Google_Maps_Huber_standardize': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Google_Maps'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,
            #                                         },
            #     'subway_in_calendar_Deezer_Huber_standardize': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Deezer'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,
            #                                         },
            #     'subway_in_calendar_Web_Weather_Huber_standardize': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,
            #                                         },
            #     'subway_in_calendar_Web_Weather_Google_Maps_Huber_standardize': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Google_Maps','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,
            #                                         },
            #     'subway_in_calendar_Web_Weather_Deezer_Huber_standardize': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Deezer','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,
            #                                         },
            #     'subway_in_calendar_Google_Maps_Deezer_Huber_standardize': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Deezer','Google_Maps'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,
            #                                         },

            #     'subway_in_calendar_Google_Maps_Deezer_Huber_standardize_expsmooth07': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Deezer','Google_Maps'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,

            #                                         'denoising_names':['netmob_POIs'],
            #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                         'denoising_modes':["train","valid","test"],             # par défaut
            #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.7}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
            #                                         },

            #     'subway_in_calendar_Web_Weather_Deezer_Huber_standardize_expsmooth07': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Deezer','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,

            #                                         'denoising_names':['netmob_POIs'],
            #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                         'denoising_modes':["train","valid","test"],             # par défaut
            #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.7}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
            #                                         },
            #   'subway_in_calendar_Web_Weather_Google_Maps_Huber_standardize_expsmooth07': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Google_Maps','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,

            #                                         'denoising_names':['netmob_POIs'],
            #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                         'denoising_modes':["train","valid","test"],             # par défaut
            #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.7}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
            #                                         },

            #     'subway_in_calendar_Google_Maps_Deezer_Huber_standardize_expsmooth08': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Deezer','Google_Maps'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,

            #                                         'denoising_names':['netmob_POIs'],
            #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                         'denoising_modes':["train","valid","test"],             # par défaut
            #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
            #                                         },

            #     'subway_in_calendar_Web_Weather_Deezer_Huber_standardize_expsmooth08': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Deezer','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,

            #                                         'denoising_names':['netmob_POIs'],
            #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                         'denoising_modes':["train","valid","test"],             # par défaut
            #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
            #                                         },
            #   'subway_in_calendar_Web_Weather_Google_Maps_Huber_standardize_expsmooth08': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Google_Maps','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,

            #                                         'denoising_names':['netmob_POIs'],
            #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                         'denoising_modes':["train","valid","test"],             # par défaut
            #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
            #                                         },

            #     'subway_in_calendar_Google_Maps_Deezer_Huber_standardize_expsmooth06': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Deezer','Google_Maps'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,

            #                                         'denoising_names':['netmob_POIs'],
            #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                         'denoising_modes':["train","valid","test"],             # par défaut
            #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.6}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
            #                                         },

            #     'subway_in_calendar_Web_Weather_Deezer_Huber_standardize_expsmooth06': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Deezer','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,

            #                                         'denoising_names':['netmob_POIs'],
            #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                         'denoising_modes':["train","valid","test"],             # par défaut
            #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.6}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
            #                                         },
            #   'subway_in_calendar_Web_Weather_Google_Maps_Huber_standardize_expsmooth06': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Google_Maps','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,

            #                                         'denoising_names':['netmob_POIs'],
            #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                         'denoising_modes':["train","valid","test"],             # par défaut
            #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.6}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
            #                                         },

            #     'subway_in_calendar_Google_Maps_Deezer_Huber_standardize_expsmooth09': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Deezer','Google_Maps'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,

            #                                         'denoising_names':['netmob_POIs'],
            #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                         'denoising_modes':["train","valid","test"],             # par défaut
            #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.9}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
            #                                         },

            #     'subway_in_calendar_Web_Weather_Deezer_Huber_standardize_expsmooth09': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Deezer','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,

            #                                         'denoising_names':['netmob_POIs'],
            #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                         'denoising_modes':["train","valid","test"],             # par défaut
            #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.9}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
            #                                         },
            #   'subway_in_calendar_Web_Weather_Google_Maps_Huber_standardize_expsmooth09': {'target_data': 'subway_in',
            #                                       'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            #                                       'embedding_calendar_types': ['dayofweek', 'hour'],
            #                                       'loss_function_type':'HuberLoss',
            #                                        'Kt': 2,
            #                                         'stblock_num': 4,
            #                                         'Ks': 2,
            #                                         'graph_conv_type': 'graph_conv',
            #                                         'gso_type': 'sym_renorm_adj',
            #                                         'enable_bias': True,
            #                                         'adj_type': 'corr',
            #                                         'enable_padding': True,
            #                                         'threshold': 0.3,
            #                                         'act_func': 'glu',
            #                                         'temporal_h_dim': 64,
            #                                         'spatial_h_dim': 256,
            #                                         'output_h_dim': 64,
            #                                         'weight_decay': 0.0014517707449388,
            #                                         'batch_size': 128,
            #                                         'lr': 0.00071,
            #                                         'dropout': 0.145169206052754,
            #                                         'epochs': 500,
            #                                         'standardize': True,
            #                                         'minmaxnorm': False,

            #                                         'NetMob_selected_apps'  :  ['Google_Maps','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
            #                                         'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                         'NetMob_selected_tags' : ['station_epsilon300'], 
            #                                         'NetMob_expanded'  : '' ,# '' # '_expanded'
            #                                         'NetMob_only_epsilon':  True ,

            #                                         'denoising_names':['netmob_POIs'],
            #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                         'denoising_modes':["train","valid","test"],             # par défaut
            #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.9}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
            #                                           },
 }

if len(constant_modif) > 0:
    modifications_bis = modifications.copy()
    modifications = {}
    for key, value in modifications_bis.items():
        modif_i = constant_modif.copy()
        modif_i.update(value)
        modifications[f"{constant_name}_{key}"] = modif_i