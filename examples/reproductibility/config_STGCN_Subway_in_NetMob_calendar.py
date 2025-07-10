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
                        'use_target_as_context': False,

                        'freq': '15min',
                        'H':6,
                        'D':1,
                        'W':0,
                        }

modifications = {



######==========================================================================
#        ONE YEAR DATA
#######========================================================================


# # 1 YEAR DATA
# # All Steps RMSE = 50.78, MAE = 25.22, MAPE = 23.16, MSE = 2592.22
#  'subway_in_calendar_emb64_out64_Huber_MinMax': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding'],
#                                                   'dataset_for_coverage': ['subway_in'],
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
#                                                     'batch_size': 256,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,
#                                                     },













# 1 YEAR DATA [Bike_in, Bike_out]
# 
'subway_in_calendar_Bike_IN_bike_OUT_emb64_out64_Huber_MinMax_agg_IRIS_100_threshold_1_attn_dim24_ff64_h2_l1_stack': {'target_data': 'subway_in',
                                                  'dataset_names': ['subway_in', 'calendar_embedding','bike_in','bike_out'],
                                                  'dataset_for_coverage': ['subway_in'],
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

                                                    'contextual_kwargs' : {'bike_in': {'compute_node_attr_with_attn':True, 
                                                                                            'stacked_contextual': True,
                                                                                            'threshold_volume_min': 1,
                                                                                            'agg_iris_target_n': 100,
                                                                                            'vision_model_name' : None,
                                                                                            'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                            'attn_kwargs': {'latent_dim': 1,
                                                                                                            'dim_feedforward' : 64,
                                                                                                            'num_heads':  2,
                                                                                                            'dim_model':  24,
                                                                                                            'keep_topk': False,
                                                                                                           },  
                                                                                            #'H' : ,
                                                                                            #'D': ,
                                                                                            #'W': , 
                                                                                },
                                                                            'bike_out': {'compute_node_attr_with_attn':True, 
                                                                                            'stacked_contextual': True,
                                                                                            'threshold_volume_min': 1,
                                                                                            'agg_iris_target_n': 100,
                                                                                            'vision_model_name' : None,
                                                                                            'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                            'attn_kwargs': {'latent_dim': 1,
                                                                                                            'dim_feedforward' : 64,
                                                                                                            'num_heads':  2,
                                                                                                            'dim_model':  24,
                                                                                                            'keep_topk': False,
                                                                                                          },  
                                                                                            #'H' : ,
                                                                                            #'D': ,
                                                                                            #'W': , 
                                                                                },
                                                                            },  
                                                    'denoising_names':[],
                                                    }, 


# 1 YEAR DATA [ Bike_out]
# 
'subway_in_calendar_bike_OUT_emb64_out64_Huber_MinMax_attn_dim48_agg_IRIS_100_threshold_1_ff128_h3_l3_stack': {'target_data': 'subway_in',
                                                  'dataset_names': ['subway_in', 'calendar_embedding','bike_out'],
                                                  'dataset_for_coverage': ['subway_in'],
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

                                                    'contextual_kwargs' : {'bike_out': {'compute_node_attr_with_attn':True, 
                                                                                            'stacked_contextual': True,
                                                                                            'threshold_volume_min': 1,
                                                                                            'agg_iris_target_n': 100,
                                                                                            'vision_model_name' : None,
                                                                                            'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                            'attn_kwargs': {'latent_dim': 1,
                                                                                                            'dim_feedforward' : 64,
                                                                                                            'num_heads':  2,
                                                                                                            'dim_model':  24,
                                                                                                            'keep_topk': False,
                                                                                                            },  
                                                                                            #'H' : ,
                                                                                            #'D': ,
                                                                                            #'W': , 
                                                                                },
                                                                            },  
                                                    'denoising_names':[],
                                                    }, 

# 1 YEAR DATA [Bike_in ]
#
'subway_in_calendar_Bike_IN_emb64_out64_Huber_MinMax_attn_dim48_agg_IRIS_100_threshold_1_ff128_h3_l3_stack': {'target_data': 'subway_in',
                                                  'dataset_names': ['subway_in', 'calendar_embedding','bike_in'],
                                                  'dataset_for_coverage': ['subway_in'],
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

                                                    'contextual_kwargs' : {'bike_in': {'compute_node_attr_with_attn':True, 
                                                                                            'stacked_contextual': True,
                                                                                            'threshold_volume_min': 1,
                                                                                            'agg_iris_target_n': 100,
                                                                                            'vision_model_name' : None,
                                                                                            'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                            'attn_kwargs': {'latent_dim': 1,
                                                                                                            'dim_feedforward' : 64,
                                                                                                            'num_heads':  2,
                                                                                                            'dim_model':  24,
                                                                                                            'keep_topk': False,
                                                                                                            },  
                                                                                            #'H' : ,
                                                                                            #'D': ,
                                                                                            #'W': , 
                                                                                },
                                                                            },  
                                                    'denoising_names':[],
                                                    }, 






















# # 1 YEAR DATA [Bike_in, Bike_out]
# # All Steps RMSE = 49.20, MAE = 24.91, MAPE = 23.47, MSE = 2431.36
# 'subway_in_calendar_Bike_IN_bike_OUT_emb64_out64_Huber_MinMax_attn_dim48_agg_IRIS_100_threshold_1_ff128_h3_l3': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','bike_in','bike_out'],
#                                                   'dataset_for_coverage': ['subway_in'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'bike_in': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'threshold_volume_min': 1,
#                                                                                             'agg_iris_target_n': 100,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 32,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2,
#                                                                                                             'dim_model':  24,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 1},  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             'bike_out': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'threshold_volume_min': 1,
#                                                                                             'agg_iris_target_n': 100,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 32,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2,
#                                                                                                             'dim_model':  24,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 1},  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # 1 YEAR DATA [ Bike_out]
# # All Steps RMSE = 50.31, MAE = 25.32, MAPE = 24.07, MSE = 2543.39
# 'subway_in_calendar_bike_OUT_emb64_out64_Huber_MinMax_attn_dim48_agg_IRIS_100_threshold_1_ff128_h3_l3': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','bike_out'],
#                                                   'dataset_for_coverage': ['subway_in'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'bike_out': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'threshold_volume_min': 1,
#                                                                                             'agg_iris_target_n': 100,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 32,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2,
#                                                                                                             'dim_model':  24,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 1},  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 

# # 1 YEAR DATA [Bike_in ]
# # All Steps RMSE = 51.18, MAE = 25.44, MAPE = 24.04, MSE = 2631.95
# 'subway_in_calendar_Bike_IN_emb64_out64_Huber_MinMax_attn_dim48_agg_IRIS_100_threshold_1_ff128_h3_l3': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','bike_in'],
#                                                   'dataset_for_coverage': ['subway_in'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'bike_in': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'threshold_volume_min': 1,
#                                                                                             'agg_iris_target_n': 100,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 32,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2,
#                                                                                                             'dim_model':  24,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 1},  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 






######==========================================================================
#        BEST MODEL OBTENU 
#######========================================================================

# # All Steps RMSE = 38.38, MAE = 21.57, MAPE = 25.75, MSE = 1474.81
# # All Steps RMSE = 38.26, MAE = 21.50, MAPE = 25.02, MSE = 1465.83
# # All Steps RMSE = 38.70, MAE = 21.52, MAPE = 24.22, MSE = 1499.26
# # All Steps RMSE = 38.57, MAE = 21.56, MAPE = 24.53, MSE = 1489.94
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


# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# #  All Steps RMSE = 38.67, MAE = 21.67, MAPE = 25.66, MSE = 1497.79
# #  All Steps RMSE = 38.86, MAE = 21.71, MAPE = 25.48, MSE = 1512.47
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l3_clustering_015_MinMax_H6_D1_concatenation_late_l_out32_output_h128': {'target_data': 'subway_in',
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
#                                                     'output_h_dim': 128,
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 32},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# #  All Steps RMSE = 38.88, MAE = 21.87, MAPE = 24.91, MSE = 1512.77
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l3_clustering_015_MinMax_H6_D1_concatenation_late_l_out64_output_h128': {'target_data': 'subway_in',
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
#                                                     'output_h_dim': 128,
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 

# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# #  All Steps RMSE = 38.67, MAE = 21.67, MAPE = 25.66, MSE = 1497.79
# # All Steps RMSE = 39.43, MAE = 22.07, MAPE = 24.61, MSE = 1556.83
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l3_clustering_015_MinMax_H6_D1_concatenation_late_l_out32': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 32},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 




# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# #  All Steps RMSE = 38.29, MAE = 21.56, MAPE = 25.36, MSE = 1467.03
# # All Steps RMSE = 38.82, MAE = 21.83, MAPE = 25.14, MSE = 1508.68
# # All Steps RMSE = 38.98, MAE = 21.81, MAPE = 25.45, MSE = 1521.20
# # All Steps RMSE = 39.16, MAE = 22.03, MAPE = 26.82, MSE = 1534.87
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l3_clustering_015_MinMax_H6_D1_concatenation_late_l_out64': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 





# # [Bike_in, Bike_out], MinMax, No Smoothing Clustering 0.15
# # All Steps RMSE = 39.68, MAE = 21.93, MAPE = 24.08, MSE = 1577.87
# # All Steps RMSE = 39.48, MAE = 21.99, MAPE = 25.68, MSE = 1560.80
# # All Steps RMSE = 38.83, MAE = 21.82, MAPE = 25.84, MSE = 1509.72
# 'subway_in_calendar_Bike_IN_bike_OUT_emb64_out64_Huber_MinMax_attn_dim48_agg_IRIS_100_threshold_1_ff128_h3_l3': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','bike_in','bike_out'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'bike_in': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'threshold_volume_min': 1,
#                                                                                             'agg_iris_target_n': 100,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 32,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2,
#                                                                                                             'dim_model':  24,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 1},  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             'bike_out': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'threshold_volume_min': 1,
#                                                                                             'agg_iris_target_n': 100,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 32,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2,
#                                                                                                             'dim_model':  24,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 1},  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # [Bike_out], MinMax, No Smoothing Clustering 0.15
# # All Steps RMSE = 40.07, MAE = 22.26, MAPE = 25.49, MSE = 1609.42
# # #ll Steps RMSE = 39.09, MAE = 21.78, MAPE = 24.51, MSE = 1530.89
# # All Steps RMSE = 39.23, MAE = 22.02, MAPE = 26.25, MSE = 1540.93
# 'subway_in_calendar_bike_OUT_emb64_out64_Huber_MinMax_attn_dim48_agg_IRIS_100_threshold_1_ff128_h3_l3': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','bike_out'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'bike_out': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'threshold_volume_min': 1,
#                                                                                             'agg_iris_target_n': 100,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 32,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2,
#                                                                                                             'dim_model':  24,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 1},  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 



# [Bike_in], MinMax, No Smoothing Clustering 0.15
# All Steps RMSE = 39.05, MAE = 22.07, MAPE = 26.04, MSE = 1527.09
# All Steps RMSE = 39.05, MAE = 21.91, MAPE = 26.02, MSE = 1527.20
# All Steps RMSE = 39.07, MAE = 21.84, MAPE = 25.81, MSE = 1527.94
# All Steps RMSE = 38.90, MAE = 21.66, MAPE = 24.98, MSE = 1515.77
# 'subway_in_calendar_Bike_IN_emb64_out64_Huber_MinMax_attn_dim48_agg_IRIS_100_threshold_1_ff128_h3_l3': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','bike_in'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'bike_in': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'threshold_volume_min': 1,
#                                                                                             'agg_iris_target_n': 100,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 32,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2,
#                                                                                                             'dim_model':  24,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 1},  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 



# # [Subway-In]
# # All Steps RMSE = 38.34, MAE = 21.38, MAPE = 24.99, MSE = 1471.33
# 'subway_in_calendar_emb64_out64_Huber_MinMax': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {},
#                                                     'denoising_names':[],
#                                                     }, 


# # [Subway-In, Subway-Out]
# # All Steps RMSE = 38.88, MAE = 21.97, MAPE = 28.35, MSE = 1514.15
# # All Steps RMSE = 39.15, MAE = 21.89, MAPE = 25.39, MSE = 1536.12
# 'subway_in_calendar_subway_OUT_emb64_out64_Huber_MinMax_stack': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','subway_out'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'subway_out': {'compute_node_attr_with_attn':False, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 




# # [Subway-In, Subway-Out]
# # All Steps RMSE = 38.63, MAE = 21.56, MAPE = 25.00, MSE = 1494.36   
# # All Steps RMSE = 38.96, MAE = 21.81, MAPE = 25.29, MSE = 1520.29
# # All Steps RMSE = 39.06, MAE = 21.91, MAPE = 26.01, MSE = 1527.79
# 'subway_in_calendar_subway_OUT_emb64_out64_Huber_MinMax_Attn_lout1_h2_d24_ff64_l64_concat': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','subway_out'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'subway_out': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, 
#                                                                                             'attn_kwargs': {'latent_dim': 64,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2,
#                                                                                                             'dim_model':  24,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 1},  
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # [Subway-In, Subway-Out]
# # All Steps RMSE = 39.13, MAE = 21.84, MAPE = 27.24, MSE = 1533.86
# 'subway_in_calendar_subway_OUT_emb64_out64_Huber_MinMax_Attn_h2_d24_ff64_l1_stack': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','subway_out'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'subway_out': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, 
#                                                                                             'attn_kwargs': {'latent_dim': 1,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2,
#                                                                                                             'dim_model':  24,
#                                                                                                             'keep_topk': False,
#                                                                                                             },  
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 







#=====================================================================================================================================
#=====================================================================================================================================




#=====================================================================================================================================
#=====================================================================================================================================

# # # [Subway-In, Subway-Out]
# # 
# 'subway_in_calendar_subway_OUT_emb64_out64_Huber_MinMax_Attn_h2_d24_ff128_l1_stack': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','subway_out'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'subway_out': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, 
#                                                                                             'attn_kwargs': {'latent_dim': 1,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  2,
#                                                                                                             'dim_model':  24,
#                                                                                                             'keep_topk': False,
#                                                                                                             },  
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # # [Subway-In, Subway-Out]
# # 
# 'subway_in_calendar_subway_OUT_emb64_out64_Huber_MinMax_Attn_h2_d24_ff32_l1_stack': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','subway_out'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'subway_out': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, 
#                                                                                             'attn_kwargs': {'latent_dim': 1,
#                                                                                                             'dim_feedforward' : 32,
#                                                                                                             'num_heads':  2,
#                                                                                                             'dim_model':  24,
#                                                                                                             'keep_topk': False,
#                                                                                                             },  
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # # [Subway-In, Subway-Out]
# # 
# 'subway_in_calendar_subway_OUT_emb64_out64_Huber_MinMax_Attn_h2_d24_ff64_l1_stack_proj7': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','subway_out'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'subway_out': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, 
#                                                                                             'attn_kwargs': {'latent_dim': 1,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2,
#                                                                                                             'dim_model':  24,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out':7},  
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 



# # # [Subway-In, Subway-Out]
# # 
# 'subway_in_calendar_subway_OUT_emb64_out64_Huber_MinMax_Attn_h2_d24_ff128_l1_stack_proj7': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','subway_out'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'subway_out': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, 
#                                                                                             'attn_kwargs': {'latent_dim': 1,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  2,
#                                                                                                             'dim_model':  24,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out':7},  
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # # [Subway-In, Subway-Out]
# # 
# 'subway_in_calendar_subway_OUT_emb64_out64_Huber_MinMax_Attn_h2_d24_ff32_l1_stack_proj7': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','subway_out'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'subway_out': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, 
#                                                                                             'attn_kwargs': {'latent_dim': 1,
#                                                                                                             'dim_feedforward' : 32,
#                                                                                                             'num_heads':  2,
#                                                                                                             'dim_model':  24,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out':7},  
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 






# #=====================================================================================================================================
# #=====================================================================================================================================













# # [Subway-In, Subway-Out]
# # All Steps RMSE = 39.41, MAE = 21.94, MAPE = 24.56, MSE = 1556.01
# # All Steps RMSE = 39.12, MAE = 21.75, MAPE = 25.52, MSE = 1534.64
# 'subway_in_calendar_subway_OUT_emb64_out64_Huber_MinMax_stack': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','subway_out'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'subway_out': {'compute_node_attr_with_attn':False, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 



# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# # All Steps RMSE = 38.52, MAE = 21.86, MAPE = 26.40, MSE = 1484.77
# # All Steps RMSE = 39.46, MAE = 22.12, MAPE = 25.48, MSE = 1559.19
# # All Steps RMSE = 39.84, MAE = 22.35, MAPE = 24.18, MSE = 1589.26
# # All Steps RMSE = 38.49, MAE = 21.89, MAPE = 25.62, MSE = 1484.01
# # All Steps RMSE = 39.40, MAE = 22.20, MAPE = 23.81, MSE = 1556.25
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l3_clustering_015_MinMax': {'target_data': 'subway_in',
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
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False},  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 



# # [Deezer], MinMax
# # All Steps RMSE = 39.01, MAE = 22.00, MAPE = 25.67, MSE = 1525.65
# # All Steps RMSE = 39.59, MAE = 22.14, MAPE = 25.60, MSE = 1570.77
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
# # All Steps RMSE = 39.88, MAE = 22.16, MAPE = 25.50, MSE = 1596.23
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





# #  [Google Maps, Deezer], Standardize
# # All Steps RMSE = 39.07, MAE = 22.07, MAPE = 26.21, MSE = 1527.94
# # All Steps RMSE = 39.46, MAE = 22.02, MAPE = 24.82, MSE = 1560.19
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


# # [Subway-In, Subway-Out]
# # All Steps RMSE = 40.60, MAE = 22.58, MAPE = 26.61, MSE = 1654.57
# 'subway_in_calendar_subway_OUT_emb64_out64_Huber_MinMax_Attn_h3_d48_ff64_l1_stack': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','subway_out'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'subway_out': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, 
#                                                                                             'attn_kwargs': {'latent_dim': 1,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  3,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             },  
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 

# # [Subway-In, Subway-Out]
# # All Steps RMSE = 39.45, MAE = 22.20, MAPE = 27.10, MSE = 1559.03
# 'subway_in_calendar_subway_OUT_emb64_out64_Huber_MinMax_Attn_h4_d48_ff64_l1_stack': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','subway_out'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'subway_out': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, 
#                                                                                             'attn_kwargs': {'latent_dim': 1,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  4,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             },  
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 




# # [Subway-In, Subway-Out]
# # All Steps RMSE = 39.94, MAE = 22.31, MAPE = 27.94, MSE = 1599.57
# 'subway_in_calendar_subway_OUT_emb64_out64_Huber_MinMax_Attn_h2_d24_ff64_l2_stack': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','subway_out'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'subway_out': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, 
#                                                                                             'attn_kwargs': {'latent_dim': 2,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  2,
#                                                                                                             'dim_model':  24,
#                                                                                                             'keep_topk': False,
#                                                                                                             },  
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # [Subway-In, Subway-Out]
# #  All Steps RMSE = 39.29, MAE = 22.19, MAPE = 27.07, MSE = 1547.16
# 'subway_in_calendar_subway_OUT_emb64_out64_Huber_MinMax_Attn_h3_d48_ff64_l2_stack': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','subway_out'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'subway_out': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, 
#                                                                                             'attn_kwargs': {'latent_dim': 2,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  3,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             },  
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 

# # [Subway-In, Subway-Out]
# # All Steps RMSE = 39.53, MAE = 22.07, MAPE = 24.39, MSE = 1565.93
# 'subway_in_calendar_subway_OUT_emb64_out64_Huber_MinMax_Attn_h4_d48_ff64_l2_stack': {'target_data': 'subway_in',
#                                                   'dataset_names': ['subway_in', 'calendar_embedding','subway_out'],
#                                                   'dataset_for_coverage': ['subway_in','netmob_POIs'],
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'subway_out': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, 
#                                                                                             'attn_kwargs': {'latent_dim': 2,
#                                                                                                             'dim_feedforward' : 64,
#                                                                                                             'num_heads':  4,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             },  
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 
# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Agg30
# # All Steps RMSE = 39.27, MAE = 22.14, MAPE = 26.01, MSE = 1544.27
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_Agg30_attn_dim48_ff128_h3_l3_MinMax_H6_D1': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'agg_iris_target_n' : 30,
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             },  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 



# # [Web_Weather], MinMax, No Smoothing  Agg50
# # All Steps RMSE = 39.31, MAE = 22.26, MAPE = 25.66, MSE = 1548.54
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Web_Weather_IRIS_Agg50_attn_dim48_ff128_h3_l3_MinMax_H6_D1': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Web_Weather'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'agg_iris_target_n' : 50,
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             },  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 

# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Agg30
# # All Steps RMSE = 38.93, MAE = 21.77, MAPE = 25.26, MSE = 1517.10
# # All Steps RMSE = 39.74, MAE = 22.08, MAPE = 25.59, MSE = 1582.79
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_Agg30_attn_dim48_ff128_h3_l3_MinMax_H6_D1_concatenation_late_l_out64': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'agg_iris_target_n' : 30,
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 



# # [Web_Weather], MinMax, No Smoothing  Agg50
# # All Steps RMSE = 38.91, MAE = 21.91, MAPE = 25.93, MSE = 1516.58
# # All Steps RMSE = 39.24, MAE = 22.08, MAPE = 26.05, MSE = 1541.42
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Web_Weather_IRIS_Agg50_attn_dim48_ff128_h3_l3_MinMax_H6_D1_concatenation_late_l_out64': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Web_Weather'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'agg_iris_target_n' : 50,
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 

# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# # All Steps RMSE = 39.80, MAE = 22.27, MAPE = 25.65, MSE = 1587.99
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_Agg50_attn_dim48_ff128_h3_l3_clustering_015_MinMax_H6_D1_concatenation_late_l_out64': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'agg_iris_target_n' : 50,
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Agg 50
# # All Steps RMSE = 39.54, MAE = 22.14, MAPE = 25.54, MSE = 1565.23
# # All Steps RMSE = 39.04, MAE = 21.98, MAPE = 26.85, MSE = 1525.58
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_Agg50_attn_dim48_ff128_h3_l3_MinMax_H6_D1_concatenation_late_l_out64': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'agg_iris_target_n' : 50,
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Agg100
# # All Steps RMSE = 39.61, MAE = 22.06, MAPE = 25.85, MSE = 1572.04
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_Agg100_attn_dim48_ff128_h3_l3_MinMax_H6_D1_concatenation_late_l_out64': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'agg_iris_target_n' : 100,
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # [Web_Weather, Google_Maps], MinMax, No Smoothing Clustering Agg 50
# # All Steps RMSE = 39.28, MAE = 21.94, MAPE = 25.27, MSE = 1544.81
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_IRIS_Agg50_attn_dim48_ff128_h3_l3_MinMax_H6_D1_concatenation_late_l_out64': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'agg_iris_target_n' : 50,
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # [Web_Weather, Google_Maps], MinMax, No Smoothing Agg 100
# # All Steps RMSE = 39.50, MAE = 22.07, MAPE = 26.76, MSE = 1561.79
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_IRIS_Agg100_attn_dim48_ff128_h3_l3_MinMax_H6_D1_concatenation_late_l_out64': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'agg_iris_target_n' : 100,
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 



# # [Web_Weather Google_Maps], MinMax, No Smoothing Agg30
# # All Steps RMSE = 39.76, MAE = 22.03, MAPE = 25.45, MSE = 1581.55
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weathe_IRIS_Agg30_attn_dim48_ff128_h3_l3_MinMax_H6_D1_concatenation_late_l_out64': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'agg_iris_target_n' : 30,
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 



# # [Web_Weather], MinMax, No Smoothing Agg100
# # All Steps RMSE = 39.28, MAE = 21.95, MAPE = 25.86, MSE = 1544.86
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Web_Weather_IRIS_Agg100_attn_dim48_ff128_h3_l3_MinMax_H6_D1_concatenation_late_l_out64': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Web_Weather'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'agg_iris_target_n' : 100,
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 



# # [Web_Weather], MinMax, No Smoothing Agg30
# # All Steps RMSE = 39.14, MAE = 21.85, MAPE = 24.39, MSE = 1534.47
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Web_Weather_IRIS_Agg30_attn_dim48_ff128_h3_l3_MinMax_H6_D1_concatenation_late_l_out64': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Web_Weather'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'agg_iris_target_n' : 30,
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 



# # [Web_Weather], MinMax, No Smoothing
# # All Steps RMSE = 39.19, MAE = 21.98, MAPE = 26.64, MSE = 1537.24
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Web_Weather_IRIS_attn_dim48_ff128_h3_l3_MinMax_clustering015_H6_D1_concatenation_late_l_out64': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Web_Weather'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'agg_iris_target_n' : None,
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # [Web_Weather], MinMax, No Smoothing Clustering 0.15
# # All Steps RMSE = 39.95, MAE = 22.31, MAPE = 24.89, MSE = 1597.97
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Web_Weather_IRIS_attn_dim48_ff128_h3_l3_MinMax_H6_D1_concatenation_late_l_out64': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Web_Weather'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': None, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'agg_iris_target_n' : None,
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# #  All Steps RMSE = 39.82, MAE = 22.04, MAPE = 24.79, MSE = 1589.24
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l3_clustering_015_MinMax_H6_D1_concatenation_late_l_out128': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 128},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 

# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# #  All Steps RMSE = 39.23, MAE = 21.94, MAPE = 25.37, MSE = 1540.00
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l2_clustering_015_MinMax_H6_D1_concatenation_late_l_out64': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 

# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# #  All Steps RMSE = 39.52, MAE = 21.82, MAPE = 24.27, MSE = 1564.68
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l6_clustering_015_MinMax_H6_D1_concatenation_late_l_out64': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 6 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 

# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# #  All Steps RMSE = 39.04, MAE = 21.77, MAPE = 25.37, MSE = 1526.44
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l4_clustering_015_MinMax_H6_D1_concatenation_late_l_out64': {'target_data': 'subway_in',
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 4 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# # All Steps RMSE = 38.86, MAE = 21.70, MAPE = 25.77, MSE = 1513.49
# #All Steps RMSE = 38.81, MAE = 21.76, MAPE = 24.78, MSE = 1507.76
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l3_clustering_015_MinMax_H6_D1_concatenation_late_l_out128': {'target_data': 'subway_in',
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
#                                                     'output_h_dim': 128,
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

                                                     
#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                      'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 6,
#                                                                                             'D': 1,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 

# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.2
# # All Steps RMSE = 39.31, MAE = 22.14, MAPE = 26.46, MSE = 1546.84
# # All Steps RMSE = 39.58, MAE = 22.33, MAPE = 25.87, MSE = 1570.34
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l3_clustering_02_MinMax': {'target_data': 'subway_in',
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
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.2, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 

# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.1
# # All Steps RMSE = 38.77, MAE = 21.92, MAPE = 25.26, MSE = 1505.38
# # All Steps RMSE = 39.66, MAE = 22.00, MAPE = 24.62, MSE = 1576.61
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l3_clustering_01_MinMax': {'target_data': 'subway_in',
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
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.1, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 



# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# # All Steps RMSE = 38.83, MAE = 21.97, MAPE = 24.95, MSE = 1510.84
# # All Steps RMSE = 40.08, MAE = 22.49, MAPE = 25.68, MSE = 1609.04
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l6_clustering_015_MinMax': {'target_data': 'subway_in',
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
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 6 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# # All Steps RMSE = 42.17, MAE = 23.86, MAPE = 29.45, MSE = 1784.82
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l6_clustering_015_MinMax_H2_concatenation_late_l_out64': {'target_data': 'subway_in',
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
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 6 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
                                                                                            
#                                                                                             'H' : 2,
#                                                                                             'D': 0 ,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 
# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# # All Steps RMSE = 43.00, MAE = 24.07, MAPE = 28.45, MSE = 1856.58
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l6_clustering_015_MinMax_H2_concatenation_late_l_out32': {'target_data': 'subway_in',
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
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 6 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 32},  
                                                                                            
#                                                                                             'H' : 2,
#                                                                                             'D': 0 ,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 



# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# # All Steps RMSE = 41.25, MAE = 22.82, MAPE = 25.52, MSE = 1705.9
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l3_clustering_015_MinMax_H2_stacked': {'target_data': 'subway_in',
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
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False} , 
#                                                                                             'H' : 2,
#                                                                                             'D': 0 ,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# # All Steps RMSE = 41.90, MAE = 23.72, MAPE = 30.74, MSE = 1763.67
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l3_clustering_015_MinMax_H2_concatenation_late': {'target_data': 'subway_in',
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
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
#                                                                                             'H' : 2,
#                                                                                             'D': 0 ,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 



# # [Web_Weather], MinMax, No Smoothing Clustering 0.15
# # All Steps RMSE = 41.02, MAE = 22.56, MAPE = 24.84, MSE = 1688.25
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Web_Weather_IRIS_attn_dim48_ff128_h3_l3_clustering_015_MinMax_H2_stacked': {'target_data': 'subway_in',
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
#                                                                                             'NetMob_selected_apps' : ['Web_Weather'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False} , 
#                                                                                             'H' : 2,
#                                                                                             'D': 0 ,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # [Web_Weather], MinMax, No Smoothing Clustering 0.15
# #  All Steps RMSE = 41.89, MAE = 23.88, MAPE = 28.94, MSE = 1761.51
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Web_Weather_IRIS_attn_dim48_ff128_h3_l3_clustering_015_MinMax_H2_concatenation_late': {'target_data': 'subway_in',
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
#                                                                                             'stacked_contextual': False,
#                                                                                             'NetMob_selected_apps' : ['Web_Weather'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out': 64},  
#                                                                                             'H' : 2,
#                                                                                             'D': 0 ,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 



# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# #  All Steps RMSE = 41.55, MAE = 22.81, MAPE = 25.11, MSE = 1734.00
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l6_clustering_015_MinMax_H2_stacked': {'target_data': 'subway_in',
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

#                                                     'optimizer': 'adamw',
#                                                     'batch_size': 128,
#                                                     'freq': '15min',
#                                                     'H':6,
#                                                     'D':1,
#                                                     'W':0,

#                                                     'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
#                                                                                             'stacked_contextual': True,
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 6 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False} , 
#                                                                                             'H' : 2,
#                                                                                             'D': 0 ,
#                                                                                             'W': 0 , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 



# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# # All Steps RMSE = 40.05, MAE = 22.70, MAPE = 26.51, MSE = 1608.17
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim64_ff128_h4_l3_clustering_015_MinMax': {'target_data': 'subway_in',
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
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 3 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  4,
#                                                                                                             'dim_model':  64,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 

# # [Web_Weather, Google_Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# # All Steps RMSE = 38.91, MAE = 21.97, MAPE = 25.70, MSE = 1516.81
# # All Steps RMSE = 38.73, MAE = 21.80, MAPE = 25.74, MSE = 1501.86
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_Deezer_IRIS_attn_dim48_ff128_h3_l2_clustering_015_MinMax': {'target_data': 'subway_in',
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
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # [Web_Weather, Google_Maps], MinMax, No Smoothing Clustering 0.15
# # All Steps RMSE = 39.54, MAE = 22.17, MAPE = 25.07, MSE = 1567.16
# # All Steps RMSE = 39.04, MAE = 21.88, MAPE = 24.69, MSE = 1526.49
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Web_Weather_IRIS_attn_dim48_ff128_h3_l2_clustering_015_MinMax': {'target_data': 'subway_in',
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
#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False}  
#                                                                                             #'H' : ,
#                                                                                             #'D': ,
#                                                                                             #'W': , 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 

# # [Google Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# # All Steps RMSE = 39.72, MAE = 22.12, MAPE = 24.72, MSE = 1584.41
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Deezer_IRIS_attn_dim48_ff128_h3_l2_clustering_015_MinMax': {'target_data': 'subway_in',
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
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False},
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
#                                                     }, 


# # [Google Maps, Deezer], MinMax, No Smoothing Clustering 0.15, H2
# # All Steps RMSE = 44.40, MAE = 25.07, MAPE = 29.85, MSE = 1986.13
# # All Steps RMSE = 41.20, MAE = 23.33, MAPE = 28.70, MSE = 1703.48
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Deezer_IRIS_attn_dim48_ff128_h3_l2_clustering_015_MinMax_H2_concatenate_late': {'target_data': 'subway_in',
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
#                                                                                             'stacked_contextual': False,

#                                                                                             'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
#                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                             'epsilon_clustering': 0.15, 
#                                                                                             'vision_model_name' : None,
#                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                             'attn_kwargs': {'latent_dim': 2 ,
#                                                                                                             'dim_feedforward' : 128,
#                                                                                                             'num_heads':  3 ,
#                                                                                                             'dim_model':  48,
#                                                                                                             'keep_topk': False,
#                                                                                                             'L_out':64},
#                                                                                             'H' : 2,
#                                                                                             'D': 0,
#                                                                                             'W': 0, 
#                                                                                 },
#                                                                             },  
#                                                     'denoising_names':[],
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
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_01': {'target_data': 'subway_in',
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

# # [Deezer], MinMax, Netmob: STD
# All Steps RMSE = 40.40, MAE = 22.61, MAPE = 27.11, MSE = 1637.98
# All Steps RMSE = 39.64, MAE = 22.26, MAPE = 27.29, MSE = 1572.82
# All Steps RMSE = 40.41, MAE = 22.30, MAPE = 25.34, MSE = 1635.67
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_01_std_netmob': {'target_data': 'subway_in',
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
#                                                                                             'minmaxnorm': False,
#                                                                                             'standardize': True,
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
# # [Deezer], MinMax, Netmob: MinMax
# All Steps RMSE = 39.37, MAE = 22.18, MAPE = 26.07, MSE = 1553.27
# All Steps RMSE = 39.33, MAE = 22.07, MAPE = 25.45, MSE = 1550.40
# All Steps RMSE = 39.31, MAE = 22.23, MAPE = 27.01, MSE = 1548.29
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Deezer_IRIS_attn_dim64_ff64_h2_l2_exp_smooth08_clustering_01_minmax_netmob': {'target_data': 'subway_in',
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
#                                                                                             'minmaxnorm': True,
#                                                                                             'standardize': False,
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


# [Google Maps, Deezer], MinMax
# All Steps RMSE = 38.96, MAE = 21.98, MAPE = 26.38, MSE = 1520.98
# All Steps RMSE = 39.06, MAE = 21.99, MAPE = 25.38, MSE = 1527.62
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

# # [Google Maps, Deezer], MinMax, No Smoothing Clustering 0.15
# # All Steps RMSE = 38.97, MAE = 22.00, MAPE = 25.87, MSE = 1520.97
# 'subway_in_calendar_emb64_out64_Huber_MinMax_Google_Maps_Deezer_IRIS_attn_dim64_ff64_h2_l2_clustering_015_MinMax': {'target_data': 'subway_in',
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
#                                                     'denoising_names':[],
#                                                     }, 



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