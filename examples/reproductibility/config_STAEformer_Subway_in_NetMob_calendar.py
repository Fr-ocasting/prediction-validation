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
                  'loss_function_type':'HuberLoss',
                  'use_target_as_context': False,
                        'optimizer': 'adamw',
                        'batch_size': 128,
                        'epochs':300,
                        'adaptive_embedding_dim': 32,
                        'input_embedding_dim': 12,
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
                        
                        #'torch_scheduler_type': 'MultiStepLR',
                        #'loss_function_type':'HuberLoss',
                        #'torch_scheduler_milestone': [25, 45, 65],
                        #'torch_scheduler_gamma':0.1,

                        'standardize': True,
                        'minmaxnorm': False,
                        'calendar_types':['dayofweek', 'timeofday']
                        }

modifications = {
######==========================================================================
#        BEST MODEL OBTENU 
#######========================================================================
                # All Steps RMSE = 40.12, MAE = 22.54, MAPE = 24.14, MSE = 1612.03
                # All Steps RMSE = 39.81, MAE = 22.44, MAPE = 25.02, MSE = 1586.97
                # All Steps RMSE = 40.00, MAE = 22.58, MAPE = 25.26, MSE = 1602.40
                #  All Steps RMSE = 40.02, MAE = 22.59, MAPE = 25.11, MSE = 1604.55
                # All Steps RMSE = 40.23, MAE = 22.67, MAPE = 24.83, MSE = 1620.54
                'calendar_input_dim24': {'dataset_names': ['subway_in','calendar'],
                                                        'input_embedding_dim': 12,
                                                        'unormalize_loss' : True,
                                                        'contextual_kwargs' : {},  
                                                        'loss_function_type':'HuberLoss',
                                                        'optimizer': 'adamw',
                                                        'batch_size': 128,
                                                        'epochs':500,
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
                                                        'calendar_types':['dayofweek', 'timeofday']
                            },

                # # [Subway-in, Subway_out]  Stack Channel
                # # All Steps RMSE = 39.92, MAE = 22.43, MAPE = 24.68, MSE = 1595.62
                # # All Steps RMSE = 40.04, MAE = 22.59, MAPE = 25.48, MSE = 1605.77
                # 'subway_in_subway_out_calendar_input_dim24_stack': {'dataset_names': ['subway_in','calendar','subway_out'],
                #                                         'input_embedding_dim': 12,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #                                         'loss_function_type':'HuberLoss',
                #                                         'optimizer': 'adamw',
                #                                         'batch_size': 128,
                #                                         'epochs':500,
                #                                         'adaptive_embedding_dim': 32,
                #                                         'input_embedding_dim': 24,
                #                                         'tod_embedding_dim': 6,
                #                                         'dow_embedding_dim': 6,
                #                                         'feed_forward_dim': 256,
                                                        
                #                                         'num_heads': 4,
                #                                         'num_layers': 3,

                #                                         'use_mixed_proj': True,
                #                                         'freq': '15min',
                #                                         'H':6,
                #                                         'D':1,
                #                                         'W':0,

                #                                         'lr': 0.001,
                #                                         'weight_decay':  0.0015,
                #                                         'dropout': 0.2,
                #                                         'torch_scheduler_milestone': 20,
                #                                         'torch_scheduler_gamma':0.9925,
                #                                         'torch_scheduler_type': 'warmup',
                #                                         'torch_scheduler_lr_start_factor': 0.3,
                #                                         'standardize': True,
                #                                         'minmaxnorm': False,
                #                                         'calendar_types':['dayofweek', 'timeofday'],
                #                                         'contextual_kwargs' : {'subway_out': {'need_global_attn':False, 
                #                                                                             'stacked_contextual': True,
                #                                                                             'vision_model_name' : None,
                #                                                                             'use_only_for_common_dates': False, 
                #                                                                             'stack_consistent_datasets' : False, # True,
                #                                                                             # 'attn_kwargs': {'latent_dim': 1,
                #                                                                             #                 'dim_feedforward' : 64,
                #                                                                             #                 'num_heads':  2,
                #                                                                             #                 'dim_model':  24,
                #                                                                             #                 'keep_topk': False,
                #                                                                             #                 },
                #                                                                       },  
                #                                                         },
                #             },


                # # [Subway-in, Subway_out] Stack Channel with ATTENTION 
                # # # All Steps RMSE = 39.56, MAE = 22.24, MAPE = 24.49, MSE = 1567.69
                # 'subway_in_subway_out_calendar_input_dim2_stack_with_attn_d48_h2_ff64_l1': {'dataset_names': ['subway_in','calendar','subway_out'],
                #                                         'input_embedding_dim': 12,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #                                         'loss_function_type':'HuberLoss',
                #                                         'optimizer': 'adamw',
                #                                         'batch_size': 128,
                #                                         'epochs':500,
                #                                         'adaptive_embedding_dim': 32,
                #                                         'input_embedding_dim': 24,
                #                                         'tod_embedding_dim': 6,
                #                                         'dow_embedding_dim': 6,
                #                                         'feed_forward_dim': 256,
                                                        
                #                                         'num_heads': 4,
                #                                         'num_layers': 3,

                #                                         'use_mixed_proj': True,
                #                                         'freq': '15min',
                #                                         'H':6,
                #                                         'D':1,
                #                                         'W':0,

                #                                         'lr': 0.001,
                #                                         'weight_decay':  0.0015,
                #                                         'dropout': 0.2,
                #                                         'torch_scheduler_milestone': 20,
                #                                         'torch_scheduler_gamma':0.9925,
                #                                         'torch_scheduler_type': 'warmup',
                #                                         'torch_scheduler_lr_start_factor': 0.3,
                #                                         'standardize': True,
                #                                         'minmaxnorm': False,
                #                                         'calendar_types':['dayofweek', 'timeofday'],
                #                                         'contextual_kwargs' : {'subway_out': {'need_global_attn':True, 
                #                                                                             'stacked_contextual': True,
                #                                                                             'vision_model_name' : None,
                #                                                                             'use_only_for_common_dates': False, 
                #                                                                             'stack_consistent_datasets' : False, # True,
                #                                                                             'attn_kwargs': {'latent_dim': 1,
                #                                                                                             'dim_feedforward' : 64,
                #                                                                                             'num_heads':  2,
                #                                                                                             'dim_model':  48,
                #                                                                                             'keep_topk': False,
                #                                                                                             },
                #                                                                       },  
                #                                                         },
                #             },

                # # All Steps RMSE = 39.33, MAE = 22.11, MAPE = 24.57, MSE = 1548.36
                # 'subway_in_subway_out_calendar_input_dim2_stack_with_attn_d48_h3_ff128_l1': {'dataset_names': ['subway_in','calendar','subway_out'],
                #                                         'input_embedding_dim': 12,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #                                         'loss_function_type':'HuberLoss',
                #                                         'optimizer': 'adamw',
                #                                         'batch_size': 128,
                #                                         'epochs':500,
                #                                         'adaptive_embedding_dim': 32,
                #                                         'input_embedding_dim': 24,
                #                                         'tod_embedding_dim': 6,
                #                                         'dow_embedding_dim': 6,
                #                                         'feed_forward_dim': 256,
                                                        
                #                                         'num_heads': 4,
                #                                         'num_layers': 3,

                #                                         'use_mixed_proj': True,
                #                                         'freq': '15min',
                #                                         'H':6,
                #                                         'D':1,
                #                                         'W':0,

                #                                         'lr': 0.001,
                #                                         'weight_decay':  0.0015,
                #                                         'dropout': 0.2,
                #                                         'torch_scheduler_milestone': 20,
                #                                         'torch_scheduler_gamma':0.9925,
                #                                         'torch_scheduler_type': 'warmup',
                #                                         'torch_scheduler_lr_start_factor': 0.3,
                #                                         'standardize': True,
                #                                         'minmaxnorm': False,
                #                                         'calendar_types':['dayofweek', 'timeofday'],
                #                                         'contextual_kwargs' : {'subway_out': {'need_global_attn':True, 
                #                                                                             'stacked_contextual': True,
                #                                                                             'vision_model_name' : None,
                #                                                                             'use_only_for_common_dates': False, 
                #                                                                             'stack_consistent_datasets' : False, # True,
                #                                                                             'attn_kwargs': {'latent_dim': 1,
                #                                                                                             'dim_feedforward' : 128,
                #                                                                                             'num_heads':  3,
                #                                                                                             'dim_model':  48,
                #                                                                                             'keep_topk': False,
                #                                                                                             },
                #                                                                       },  
                #                                                         },
                #             },   
                                                        



    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # AGGREGATE IRIS ZONE TO REDUCE DIMENSIONALITY
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#      # Agg IRIS: choices = [30 - 50 - 100]                                       
#       # All Steps RMSE = 40.45, MAE = 22.82, MAPE = 25.61, MSE = 1639.05
#       'calendar_Google_Maps_Deezer_IRIS_agg_30_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering02': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                                 'loss_function_type':'HuberLoss',
#                                                 'optimizer': 'adamw',
#                                                 'batch_size': 128,
#                                                 'epochs':500,
#                                                 'adaptive_embedding_dim': 32,
#                                                 'tod_embedding_dim': 6,
#                                                 'dow_embedding_dim': 6,
#                                                 'feed_forward_dim': 256,
#                                                 'input_embedding_dim': 24,
#                                                 'num_heads': 4,
#                                                 'num_layers': 3,
#                                                 'lr': 0.001,
#                                                 'weight_decay':  0.0015,
#                                                 'dropout': 0.2,
#                                                 'torch_scheduler_milestone': 20,
#                                                 'torch_scheduler_gamma':0.9925,
#                                                 'torch_scheduler_type': 'warmup',
#                                                 'torch_scheduler_lr_start_factor': 0.3,

#                                                 'use_mixed_proj': True,
#                                                 'freq': '15min',
#                                                 'H':6,
#                                                 'D':1,
#                                                 'W':0,

#                                                 'standardize': True,
#                                                 'minmaxnorm': False,
#                                                 'calendar_types':['dayofweek', 'timeofday'],

#                                                 'unormalize_loss' : True,
#                                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                         'stacked_contextual': True,
#                                                                                          'agg_iris_target_n': 30,
#                                                                                         'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
#                                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                         'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                         'vision_model_name' : None,
#                                                                                         'epsilon_clustering': 0.2,
#                                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                         'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                         'dim_feedforward' : 64,
#                                                                                                         'num_heads' : 2 ,
#                                                                                                         'dim_model' : 12,
#                                                                                                         'keep_topk': False}  
#                                                                                         #'H' : ,
#                                                                                         #'D': ,
#                                                                                         #'W': , 
#                                                                             },
#                                                                     },  
#                                                 'denoising_names':['netmob_POIs'],
#                                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                 'denoising_modes':["train","valid","test"],             # par défaut
#                                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                     },

#         # All Steps RMSE = 40.14, MAE = 22.58, MAPE = 25.14, MSE = 1613.58
#        'calendar_Google_Maps_IRIS_agg_30_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering02': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.2,
#                                                                                 'agg_iris_target_n': 30,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },


#         # All Steps RMSE = 40.19, MAE = 22.99, MAPE = 26.16, MSE = 1617.00
#        'calendar_Deezer_IRIS_agg_30_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 30,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },

                                   
#       # All Steps RMSE = 40.07, MAE = 22.77, MAPE = 24.96, MSE = 1607.61
#       'calendar_Google_Maps_Deezer_IRIS_agg_50_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering02': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                                 'loss_function_type':'HuberLoss',
#                                                 'optimizer': 'adamw',
#                                                 'batch_size': 128,
#                                                 'epochs':500,
#                                                 'adaptive_embedding_dim': 32,
#                                                 'tod_embedding_dim': 6,
#                                                 'dow_embedding_dim': 6,
#                                                 'feed_forward_dim': 256,
#                                                 'input_embedding_dim': 24,
#                                                 'num_heads': 4,
#                                                 'num_layers': 3,
#                                                 'lr': 0.001,
#                                                 'weight_decay':  0.0015,
#                                                 'dropout': 0.2,
#                                                 'torch_scheduler_milestone': 20,
#                                                 'torch_scheduler_gamma':0.9925,
#                                                 'torch_scheduler_type': 'warmup',
#                                                 'torch_scheduler_lr_start_factor': 0.3,

#                                                 'use_mixed_proj': True,
#                                                 'freq': '15min',
#                                                 'H':6,
#                                                 'D':1,
#                                                 'W':0,

#                                                 'standardize': True,
#                                                 'minmaxnorm': False,
#                                                 'calendar_types':['dayofweek', 'timeofday'],

#                                                 'unormalize_loss' : True,
#                                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                         'stacked_contextual': True,
#                                                                                          'agg_iris_target_n': 50,
#                                                                                         'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
#                                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                         'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                         'vision_model_name' : None,
#                                                                                         'epsilon_clustering': 0.2,
#                                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                         'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                         'dim_feedforward' : 64,
#                                                                                                         'num_heads' : 2 ,
#                                                                                                         'dim_model' : 12,
#                                                                                                         'keep_topk': False}  
#                                                                                         #'H' : ,
#                                                                                         #'D': ,
#                                                                                         #'W': , 
#                                                                             },
#                                                                     },  
#                                                 'denoising_names':['netmob_POIs'],
#                                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                 'denoising_modes':["train","valid","test"],             # par défaut
#                                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                     },

#         # All Steps RMSE = 40.12, MAE = 22.79, MAPE = 25.73, MSE = 1612.51
#        'calendar_Google_Maps_IRIS_agg_50_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering02': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.2,
#                                                                                 'agg_iris_target_n': 50,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },


#         # All Steps RMSE = 39.92, MAE = 22.68, MAPE = 25.09, MSE = 1595.01
#        'calendar_Deezer_IRIS_agg_50_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 50,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },

                                    
#       # All Steps RMSE = 40.81, MAE = 23.18, MAPE = 26.43, MSE = 1670.03
#       'calendar_Google_Maps_Deezer_IRIS_agg_100_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering02': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                                 'loss_function_type':'HuberLoss',
#                                                 'optimizer': 'adamw',
#                                                 'batch_size': 128,
#                                                 'epochs':500,
#                                                 'adaptive_embedding_dim': 32,
#                                                 'tod_embedding_dim': 6,
#                                                 'dow_embedding_dim': 6,
#                                                 'feed_forward_dim': 256,
#                                                 'input_embedding_dim': 24,
#                                                 'num_heads': 4,
#                                                 'num_layers': 3,
#                                                 'lr': 0.001,
#                                                 'weight_decay':  0.0015,
#                                                 'dropout': 0.2,
#                                                 'torch_scheduler_milestone': 20,
#                                                 'torch_scheduler_gamma':0.9925,
#                                                 'torch_scheduler_type': 'warmup',
#                                                 'torch_scheduler_lr_start_factor': 0.3,

#                                                 'use_mixed_proj': True,
#                                                 'freq': '15min',
#                                                 'H':6,
#                                                 'D':1,
#                                                 'W':0,

#                                                 'standardize': True,
#                                                 'minmaxnorm': False,
#                                                 'calendar_types':['dayofweek', 'timeofday'],

#                                                 'unormalize_loss' : True,
#                                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                         'stacked_contextual': True,
#                                                                                          'agg_iris_target_n': 100,
#                                                                                         'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
#                                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                         'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                         'vision_model_name' : None,
#                                                                                         'epsilon_clustering': 0.2,
#                                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                         'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                         'dim_feedforward' : 64,
#                                                                                                         'num_heads' : 2 ,
#                                                                                                         'dim_model' : 12,
#                                                                                                         'keep_topk': False}  
#                                                                                         #'H' : ,
#                                                                                         #'D': ,
#                                                                                         #'W': , 
#                                                                             },
#                                                                     },  
#                                                 'denoising_names':['netmob_POIs'],
#                                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                                 'denoising_modes':["train","valid","test"],             # par défaut
#                                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#                     },

#         # All Steps RMSE = 39.79, MAE = 22.57, MAPE = 25.09, MSE = 1586.30
#        'calendar_Google_Maps_IRIS_agg_100_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering02': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.2,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },


#         # All Steps RMSE = 39.66, MAE = 22.45, MAPE = 26.71, MSE = 1574.70
#        'calendar_Deezer_IRIS_agg_100_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },


#         # All Steps RMSE = 40.79, MAE = 23.33, MAPE = 26.44, MSE = 1667.21
#        'calendar_Google_Maps_Deezer_IRIS_agg_100_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },

#         # All Steps RMSE = 40.04, MAE = 22.59, MAPE = 25.09, MSE = 1605.39
#        'calendar_Google_Maps_Deezer_IRIS_agg_100_input_dim24_attn_dim64_ff64_h2_ldim4_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 4 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },

#         # All Steps RMSE = 40.01, MAE = 22.61, MAPE = 24.81, MSE = 1603.77
#        'calendar_Web_Weather_Deezer_IRIS_agg_100_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Deezer','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },

#         # All Steps RMSE = 39.61, MAE = 22.45, MAPE = 25.11, MSE = 1571.09
#        'calendar_Web_Weather_Deezer_IRIS_agg_100_input_dim24_attn_dim64_ff64_h2_ldim4_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Deezer','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 4 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },


        # All Steps RMSE = 38.98, MAE = 22.32, MAPE = 26.05, MSE = 1520.75
        # All Steps RMSE = 39.14, MAE = 22.16, MAPE = 26.02, MSE = 1533.68
        #  All Steps RMSE = 39.75, MAE = 22.45, MAPE = 25.33, MSE = 1581.65
        # All Steps RMSE = 39.09, MAE = 22.18, MAPE = 26.25, MSE = 1529.74
        # All Steps RMSE = 40.00, MAE = 22.64, MAPE = 25.57, MSE = 1602.63
        # All Steps RMSE = 39.41, MAE = 22.25, MAPE = 25.92, MSE = 1554.86
        # All Steps RMSE = 39.10, MAE = 22.04, MAPE = 25.24, MSE = 1530.37
       'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim12_ff64_h2_ldim2_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                        'loss_function_type':'HuberLoss',
                                        'optimizer': 'adamw',
                                        'batch_size': 128,
                                        'epochs':500,
                                        'adaptive_embedding_dim': 32,
                                        'tod_embedding_dim': 6,
                                        'dow_embedding_dim': 6,
                                        'feed_forward_dim': 256,
                                        'input_embedding_dim': 24,
                                        'num_heads': 4,
                                        'num_layers': 3,
                                        'lr': 0.001,
                                        'weight_decay':  0.0015,
                                        'dropout': 0.2,
                                        'torch_scheduler_milestone': 20,
                                        'torch_scheduler_gamma':0.9925,
                                        'torch_scheduler_type': 'warmup',
                                        'torch_scheduler_lr_start_factor': 0.3,

                                        'use_mixed_proj': True,
                                        'freq': '15min',
                                        'H':6,
                                        'D':1,
                                        'W':0,

                                        'standardize': True,
                                        'minmaxnorm': False,
                                        'calendar_types':['dayofweek', 'timeofday'],

                                        'unormalize_loss' : True,
                                        'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                                                                                'stacked_contextual': True,
                                                                                'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
                                                                                'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                'NetMob_expanded' : '', # '' # '_expanded'
                                                                                'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                'vision_model_name' : None,
                                                                                'epsilon_clustering': 0.1,
                                                                                'agg_iris_target_n': 100,
                                                                                'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                'attn_kwargs': {'latent_dim' : 2 ,
                                                                                                'dim_feedforward' : 64,
                                                                                                'num_heads' : 2 ,
                                                                                                'dim_model' : 12,
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
#       # All Steps RMSE = 39.33, MAE = 22.27, MAPE = 24.76, MSE = 1548.55
#         # All Steps RMSE = 39.46, MAE = 22.30, MAPE = 24.31, MSE = 1558.68
#         # All Steps RMSE = 39.49, MAE = 22.39, MAPE = 24.76, MSE = 1561.84
#         # All Steps RMSE = 40.12, MAE = 22.72, MAPE = 26.45, MSE = 1612.39
#         # All Steps RMSE = 40.80, MAE = 23.20, MAPE = 26.96, MSE = 1667.70
#         'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim12_ff64_h2_ldim1_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 1 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },

#         # All Steps RMSE = 38.94, MAE = 22.00, MAPE = 24.77, MSE = 1517.79
#         # All Steps RMSE = 39.68, MAE = 22.46, MAPE = 26.24, MSE = 1576.60
#         # All Steps RMSE = 40.66, MAE = 23.15, MAPE = 26.11, MSE = 1655.73
#         # All Steps RMSE = 40.11, MAE = 22.81, MAPE = 25.20, MSE = 1611.25
#         #  All Steps RMSE = 40.26, MAE = 22.89, MAPE = 24.92, MSE = 1623.78
#         'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim12_ff128_h2_ldim1_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 1 ,
#                                                                                                 'dim_feedforward' : 128,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },
  
        # # All Steps RMSE = 39.78, MAE = 22.56, MAPE = 26.06, MSE = 1585.63
        # 'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim12_ff128_h2_ldim2_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
        #                                 'loss_function_type':'HuberLoss',
        #                                 'optimizer': 'adamw',
        #                                 'batch_size': 128,
        #                                 'epochs':500,
        #                                 'adaptive_embedding_dim': 32,
        #                                 'tod_embedding_dim': 6,
        #                                 'dow_embedding_dim': 6,
        #                                 'feed_forward_dim': 256,
        #                                 'input_embedding_dim': 24,
        #                                 'num_heads': 4,
        #                                 'num_layers': 3,
        #                                 'lr': 0.001,
        #                                 'weight_decay':  0.0015,
        #                                 'dropout': 0.2,
        #                                 'torch_scheduler_milestone': 20,
        #                                 'torch_scheduler_gamma':0.9925,
        #                                 'torch_scheduler_type': 'warmup',
        #                                 'torch_scheduler_lr_start_factor': 0.3,

        #                                 'use_mixed_proj': True,
        #                                 'freq': '15min',
        #                                 'H':6,
        #                                 'D':1,
        #                                 'W':0,

        #                                 'standardize': True,
        #                                 'minmaxnorm': False,
        #                                 'calendar_types':['dayofweek', 'timeofday'],

        #                                 'unormalize_loss' : True,
        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
        #                                                                         'stacked_contextual': True,
        #                                                                         'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
        #                                                                         'vision_model_name' : None,
        #                                                                         'epsilon_clustering': 0.1,
        #                                                                         'agg_iris_target_n': 100,
        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
        #                                                                         'attn_kwargs': {'latent_dim' : 2 ,
        #                                                                                         'dim_feedforward' : 128,
        #                                                                                         'num_heads' : 2 ,
        #                                                                                         'dim_model' : 12,
        #                                                                                         'keep_topk': False}  
        #                                                                         #'H' : ,
        #                                                                         #'D': ,
        #                                                                         #'W': , 
        #                                                             },
        #                                                     },  
        #                                 'denoising_names':['netmob_POIs'],
        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        #                                 'denoising_modes':["train","valid","test"],             # par défaut
        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
        #     },
  
#      # All Steps RMSE = 38.86, MAE = 22.07, MAPE = 24.95, MSE = 1512.00
#      # All Steps RMSE = 39.69, MAE = 22.43, MAPE = 24.97, MSE = 1577.15
#      # All Steps RMSE = 39.85, MAE = 22.58, MAPE = 25.00, MSE = 1590.28
#      # All Steps RMSE = 39.80, MAE = 22.66, MAPE = 26.09, MSE = 1586.45
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim12_ff64_h2_ldim2_exp_smooth_07_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.7}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },


#      # All Steps RMSE = 39.26, MAE = 22.33, MAPE = 24.97, MSE = 1543.35
#      # All Steps RMSE = 40.24, MAE = 22.85, MAPE = 26.76, MSE = 1622.94
#      # All Steps RMSE = 39.95, MAE = 22.64, MAPE = 26.61, MSE = 1598.88
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim36_ff64_h2_ldim2_exp_smooth_07_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 36,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.7}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },


#      # All Steps RMSE = 39.11, MAE = 22.19, MAPE = 24.81, MSE = 1531.53
#      # All Steps RMSE = 40.55, MAE = 22.85, MAPE = 26.57, MSE = 1649.42
#      # All Steps RMSE = 40.49, MAE = 22.95, MAPE = 27.83, MSE = 1644.65
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim48_ff64_h2_ldim2_exp_smooth_07_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 48,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.7}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },

#      # All Steps RMSE = 39.60, MAE = 22.65, MAPE = 27.47, MSE = 1570.85
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim24_ff64_h2_ldim2_exp_smooth_07_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 24,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.7}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },



#      # All Steps RMSE = 39.91, MAE = 22.88, MAPE = 25.50, MSE = 1596.77
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim48_ff64_h3_ldim2_exp_smooth_07_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 3 ,
#                                                                                                 'dim_model' : 48,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.7}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },


#      # All Steps RMSE = 39.35, MAE = 22.37, MAPE = 25.17, MSE = 1550.87
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim36_ff64_h3_ldim2_exp_smooth_07_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 3 ,
#                                                                                                 'dim_model' : 36,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.7}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },

#      # All Steps RMSE = 39.07, MAE = 22.26, MAPE = 24.53, MSE = 1528.85
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim48_ff64_h2_ldim2_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 48,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },

#         # All Steps RMSE = 39.73, MAE = 22.41, MAPE = 24.96, MSE = 1580.48
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim12_ff64_h2_ldim2_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': None,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },


#      # All Steps RMSE = 40.29, MAE = 22.83, MAPE = 24.64, MSE = 1625.64
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim12_ff64_h2_ldim2_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':[],
#             },


#      # All Steps RMSE = 39.29, MAE = 22.36, MAPE = 25.57, MSE = 1546.42
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim12_ff64_h2_ldim2': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': None,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':[],
#             },



#      # All Steps RMSE = 39.85, MAE = 22.65, MAPE = 25.95, MSE = 1590.69
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim12_ff64_h2_ldim2_exp_smooth_09_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.9}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },

#      # All Steps RMSE = 39.29, MAE = 22.33, MAPE = 25.86, MSE = 1545.16
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim12_ff64_h2_ldim2_exp_smooth_08_clustering015': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.15,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },


#      # All Steps RMSE = 39.67, MAE = 22.58, MAPE = 24.67, MSE = 1577.24
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim12_ff64_h2_ldim2_exp_smooth_08_clustering02': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.2,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },


#      # All Steps RMSE = 40.78, MAE = 23.17, MAPE = 27.29, MSE = 1666.88
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim12_ff128_h2_ldim2_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 128,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },

#      # All Steps RMSE = 40.19, MAE = 22.80, MAPE = 25.61, MSE = 1616.97
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim12_ff128_h2_ldim4_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 4 ,
#                                                                                                 'dim_feedforward' : 128,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },


#      # All Steps RMSE = 39.39, MAE = 22.65, MAPE = 28.00, MSE = 1553.85
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim24_ff64_h2_ldim2_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 24,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },



#      # All Steps RMSE = 40.31, MAE = 22.99, MAPE = 25.64, MSE = 1630.39
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 64,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },


#      # All Steps RMSE = 39.54, MAE = 22.40, MAPE = 25.50, MSE = 1565.59
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim64_ff64_h4_ldim2_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 4 ,
#                                                                                                 'dim_model' : 64,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },


#      # All Steps RMSE = 39.79, MAE = 22.67, MAPE = 26.54, MSE = 1586.25
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim64_ff64_h4_ldim4_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 4 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 4 ,
#                                                                                                 'dim_model' : 64,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },
#         # All Steps RMSE = 40.27, MAE = 22.73, MAPE = 25.14, MSE = 1624.62
#        'calendar_Web_Weather_Google_Maps_IRIS_agg_100_input_dim24_attn_dim64_ff64_h2_ldim4_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 4 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },


#         # All Steps RMSE = 39.72, MAE = 22.52, MAPE = 25.82, MSE = 1580.49
#        'calendar_Web_Weather_Google_Maps_Deezer_IRIS_agg_100_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Deezer','Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },

#         # All Steps RMSE = 40.47, MAE = 23.18, MAPE = 26.52, MSE = 1642.17
#        'calendar_Web_Weather_Google_Maps_Deezer_IRIS_agg_100_input_dim24_attn_dim64_ff64_h2_ldim4_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Deezer','Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 4 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },

#         # All Steps RMSE = 40.40, MAE = 22.72, MAPE = 26.09, MSE = 1634.76
#        'calendar_Web_Weather_Google_Maps_Deezer_IRIS_agg_100_input_dim24_attn_dim64_ff64_h2_ldim6_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Deezer','Google_Maps','Web_Weather'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 6 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },
#         # All Steps RMSE = 40.62, MAE = 22.92, MAPE = 26.23, MSE = 1653.52
#        'calendar_Deezer_IRIS_agg_100_input_dim24_attn_dim64_ff64_h2_ldim2_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': 0.1,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':[],
#             },



#         # All Steps RMSE = 40.04, MAE = 22.62, MAPE = 25.06, MSE = 1605.22
#        'calendar_Deezer_IRIS_agg_100_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': None,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':['netmob_POIs'],
#                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
#                                         'denoising_modes':["train","valid","test"],             # par défaut
#                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
#             },


#         # All Steps RMSE = 40.78, MAE = 23.00, MAPE = 24.48, MSE = 1664.74
#        'calendar_Deezer_IRIS_agg_100_input_dim24_attn_dim64_ff64_h2_ldim2': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
#                                         'loss_function_type':'HuberLoss',
#                                         'optimizer': 'adamw',
#                                         'batch_size': 128,
#                                         'epochs':500,
#                                         'adaptive_embedding_dim': 32,
#                                         'tod_embedding_dim': 6,
#                                         'dow_embedding_dim': 6,
#                                         'feed_forward_dim': 256,
#                                         'input_embedding_dim': 24,
#                                         'num_heads': 4,
#                                         'num_layers': 3,
#                                         'lr': 0.001,
#                                         'weight_decay':  0.0015,
#                                         'dropout': 0.2,
#                                         'torch_scheduler_milestone': 20,
#                                         'torch_scheduler_gamma':0.9925,
#                                         'torch_scheduler_type': 'warmup',
#                                         'torch_scheduler_lr_start_factor': 0.3,

#                                         'use_mixed_proj': True,
#                                         'freq': '15min',
#                                         'H':6,
#                                         'D':1,
#                                         'W':0,

#                                         'standardize': True,
#                                         'minmaxnorm': False,
#                                         'calendar_types':['dayofweek', 'timeofday'],

#                                         'unormalize_loss' : True,
#                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
#                                                                                 'stacked_contextual': True,
#                                                                                 'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
#                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
#                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
#                                                                                 'vision_model_name' : None,
#                                                                                 'epsilon_clustering': None,
#                                                                                 'agg_iris_target_n': 100,
#                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
#                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
#                                                                                                 'dim_feedforward' : 64,
#                                                                                                 'num_heads' : 2 ,
#                                                                                                 'dim_model' : 12,
#                                                                                                 'keep_topk': False}  
#                                                                                 #'H' : ,
#                                                                                 #'D': ,
#                                                                                 #'W': , 
#                                                                     },
#                                                             },  
#                                         'denoising_names':[],
#             },
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # END AGGREGATE IRIS ZONE
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


                # # [Subway-in, Subway_out] Stack Channel with ATTENTION 
                # # # All Steps RMSE = 40.24, MAE = 22.54, MAPE = 25.63, MSE = 1622.40
                # 'subway_in_subway_out_calendar_input_dim2_stack_with_attn_d24_h2_ff64_l1': {'dataset_names': ['subway_in','calendar','subway_out'],
                #                                         'input_embedding_dim': 12,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #                                         'loss_function_type':'HuberLoss',
                #                                         'optimizer': 'adamw',
                #                                         'batch_size': 128,
                #                                         'epochs':500,
                #                                         'adaptive_embedding_dim': 32,
                #                                         'input_embedding_dim': 24,
                #                                         'tod_embedding_dim': 6,
                #                                         'dow_embedding_dim': 6,
                #                                         'feed_forward_dim': 256,
                                                        
                #                                         'num_heads': 4,
                #                                         'num_layers': 3,

                #                                         'use_mixed_proj': True,
                #                                         'freq': '15min',
                #                                         'H':6,
                #                                         'D':1,
                #                                         'W':0,

                #                                         'lr': 0.001,
                #                                         'weight_decay':  0.0015,
                #                                         'dropout': 0.2,
                #                                         'torch_scheduler_milestone': 20,
                #                                         'torch_scheduler_gamma':0.9925,
                #                                         'torch_scheduler_type': 'warmup',
                #                                         'torch_scheduler_lr_start_factor': 0.3,
                #                                         'standardize': True,
                #                                         'minmaxnorm': False,
                #                                         'calendar_types':['dayofweek', 'timeofday'],
                #                                         'contextual_kwargs' : {'subway_out': {'need_global_attn':True, 
                #                                                                             'stacked_contextual': True,
                #                                                                             'vision_model_name' : None,
                #                                                                             'use_only_for_common_dates': False, 
                #                                                                             'stack_consistent_datasets' : False, # True,
                #                                                                             'attn_kwargs': {'latent_dim': 1,
                #                                                                                             'dim_feedforward' : 64,
                #                                                                                             'num_heads':  2,
                #                                                                                             'dim_model':  24,
                #                                                                                             'keep_topk': False,
                #                                                                                             },
                #                                                                       },  
                #                                                         },
                #             },

                # # [Subway-in, Subway_out] Stack Channel with ATTENTION 
                # # # All Steps RMSE = 39.96, MAE = 22.51, MAPE = 24.54, MSE = 1599.13
                # 'subway_in_subway_out_calendar_input_dim2_stack_with_attn_d64_h2_ff64_l1': {'dataset_names': ['subway_in','calendar','subway_out'],
                #                                         'input_embedding_dim': 12,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #                                         'loss_function_type':'HuberLoss',
                #                                         'optimizer': 'adamw',
                #                                         'batch_size': 128,
                #                                         'epochs':500,
                #                                         'adaptive_embedding_dim': 32,
                #                                         'input_embedding_dim': 24,
                #                                         'tod_embedding_dim': 6,
                #                                         'dow_embedding_dim': 6,
                #                                         'feed_forward_dim': 256,
                                                        
                #                                         'num_heads': 4,
                #                                         'num_layers': 3,

                #                                         'use_mixed_proj': True,
                #                                         'freq': '15min',
                #                                         'H':6,
                #                                         'D':1,
                #                                         'W':0,

                #                                         'lr': 0.001,
                #                                         'weight_decay':  0.0015,
                #                                         'dropout': 0.2,
                #                                         'torch_scheduler_milestone': 20,
                #                                         'torch_scheduler_gamma':0.9925,
                #                                         'torch_scheduler_type': 'warmup',
                #                                         'torch_scheduler_lr_start_factor': 0.3,
                #                                         'standardize': True,
                #                                         'minmaxnorm': False,
                #                                         'calendar_types':['dayofweek', 'timeofday'],
                #                                         'contextual_kwargs' : {'subway_out': {'need_global_attn':True, 
                #                                                                             'stacked_contextual': True,
                #                                                                             'vision_model_name' : None,
                #                                                                             'use_only_for_common_dates': False, 
                #                                                                             'stack_consistent_datasets' : False, # True,
                #                                                                             'attn_kwargs': {'latent_dim': 1,
                #                                                                                             'dim_feedforward' : 64,
                #                                                                                             'num_heads':  2,
                #                                                                                             'dim_model':  64,
                #                                                                                             'keep_topk': False,
                #                                                                                             },
                #                                                                       },  
                #                                                         },
                #             },

                # # [Subway-in, Subway_out] Stack Channel with ATTENTION 
                # # # All Steps RMSE = 39.85, MAE = 22.47, MAPE = 24.25, MSE = 1590.44
                # 'subway_in_subway_out_calendar_input_dim2_stack_with_attn_d64_h4_ff64_l1': {'dataset_names': ['subway_in','calendar','subway_out'],
                #                                         'input_embedding_dim': 12,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #                                         'loss_function_type':'HuberLoss',
                #                                         'optimizer': 'adamw',
                #                                         'batch_size': 128,
                #                                         'epochs':500,
                #                                         'adaptive_embedding_dim': 32,
                #                                         'input_embedding_dim': 24,
                #                                         'tod_embedding_dim': 6,
                #                                         'dow_embedding_dim': 6,
                #                                         'feed_forward_dim': 256,
                                                        
                #                                         'num_heads': 4,
                #                                         'num_layers': 3,

                #                                         'use_mixed_proj': True,
                #                                         'freq': '15min',
                #                                         'H':6,
                #                                         'D':1,
                #                                         'W':0,

                #                                         'lr': 0.001,
                #                                         'weight_decay':  0.0015,
                #                                         'dropout': 0.2,
                #                                         'torch_scheduler_milestone': 20,
                #                                         'torch_scheduler_gamma':0.9925,
                #                                         'torch_scheduler_type': 'warmup',
                #                                         'torch_scheduler_lr_start_factor': 0.3,
                #                                         'standardize': True,
                #                                         'minmaxnorm': False,
                #                                         'calendar_types':['dayofweek', 'timeofday'],
                #                                         'contextual_kwargs' : {'subway_out': {'need_global_attn':True, 
                #                                                                             'stacked_contextual': True,
                #                                                                             'vision_model_name' : None,
                #                                                                             'use_only_for_common_dates': False, 
                #                                                                             'stack_consistent_datasets' : False, # True,
                #                                                                             'attn_kwargs': {'latent_dim': 1,
                #                                                                                             'dim_feedforward' : 64,
                #                                                                                             'num_heads':  4,
                #                                                                                             'dim_model':  64,
                #                                                                                             'keep_topk': False,
                #                                                                                             },
                #                                                                       },  
                #                                                         },
                #             },    
                # # [Subway-in, Subway_out] Stack Channel with ATTENTION 
                # # # All Steps RMSE = 39.78, MAE = 22.37, MAPE = 25.04, MSE = 1583.91
                # 'subway_in_subway_out_calendar_input_dim2_stack_with_attn_d128_h4_ff64_l1': {'dataset_names': ['subway_in','calendar','subway_out'],
                #                                         'input_embedding_dim': 12,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #                                         'loss_function_type':'HuberLoss',
                #                                         'optimizer': 'adamw',
                #                                         'batch_size': 128,
                #                                         'epochs':500,
                #                                         'adaptive_embedding_dim': 32,
                #                                         'input_embedding_dim': 24,
                #                                         'tod_embedding_dim': 6,
                #                                         'dow_embedding_dim': 6,
                #                                         'feed_forward_dim': 256,
                                                        
                #                                         'num_heads': 4,
                #                                         'num_layers': 3,

                #                                         'use_mixed_proj': True,
                #                                         'freq': '15min',
                #                                         'H':6,
                #                                         'D':1,
                #                                         'W':0,

                #                                         'lr': 0.001,
                #                                         'weight_decay':  0.0015,
                #                                         'dropout': 0.2,
                #                                         'torch_scheduler_milestone': 20,
                #                                         'torch_scheduler_gamma':0.9925,
                #                                         'torch_scheduler_type': 'warmup',
                #                                         'torch_scheduler_lr_start_factor': 0.3,
                #                                         'standardize': True,
                #                                         'minmaxnorm': False,
                #                                         'calendar_types':['dayofweek', 'timeofday'],
                #                                         'contextual_kwargs' : {'subway_out': {'need_global_attn':True, 
                #                                                                             'stacked_contextual': True,
                #                                                                             'vision_model_name' : None,
                #                                                                             'use_only_for_common_dates': False, 
                #                                                                             'stack_consistent_datasets' : False, # True,
                #                                                                             'attn_kwargs': {'latent_dim': 1,
                #                                                                                             'dim_feedforward' : 64,
                #                                                                                             'num_heads':  4,
                #                                                                                             'dim_model':  128,
                #                                                                                             'keep_topk': False,
                #                                                                                             },
                #                                                                       },  
                #                                                         },
                #             },  

                # # # All Steps RMSE = 39.95, MAE = 22.49, MAPE = 25.93, MSE = 1599.01
                # 'subway_in_subway_out_calendar_input_dim2_stack_with_attn_d128_h4_ff128_l1': {'dataset_names': ['subway_in','calendar','subway_out'],
                #                                         'input_embedding_dim': 12,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #                                         'loss_function_type':'HuberLoss',
                #                                         'optimizer': 'adamw',
                #                                         'batch_size': 128,
                #                                         'epochs':500,
                #                                         'adaptive_embedding_dim': 32,
                #                                         'input_embedding_dim': 24,
                #                                         'tod_embedding_dim': 6,
                #                                         'dow_embedding_dim': 6,
                #                                         'feed_forward_dim': 256,
                                                        
                #                                         'num_heads': 4,
                #                                         'num_layers': 3,

                #                                         'use_mixed_proj': True,
                #                                         'freq': '15min',
                #                                         'H':6,
                #                                         'D':1,
                #                                         'W':0,

                #                                         'lr': 0.001,
                #                                         'weight_decay':  0.0015,
                #                                         'dropout': 0.2,
                #                                         'torch_scheduler_milestone': 20,
                #                                         'torch_scheduler_gamma':0.9925,
                #                                         'torch_scheduler_type': 'warmup',
                #                                         'torch_scheduler_lr_start_factor': 0.3,
                #                                         'standardize': True,
                #                                         'minmaxnorm': False,
                #                                         'calendar_types':['dayofweek', 'timeofday'],
                #                                         'contextual_kwargs' : {'subway_out': {'need_global_attn':True, 
                #                                                                             'stacked_contextual': True,
                #                                                                             'vision_model_name' : None,
                #                                                                             'use_only_for_common_dates': False, 
                #                                                                             'stack_consistent_datasets' : False, # True,
                #                                                                             'attn_kwargs': {'latent_dim': 1,
                #                                                                                             'dim_feedforward' : 128,
                #                                                                                             'num_heads':  4,
                #                                                                                             'dim_model':  128,
                #                                                                                             'keep_topk': False,
                #                                                                                             },
                #                                                                       },  
                #                                                         },
                #             },   

                # # All Steps RMSE = 39.85, MAE = 22.41, MAPE = 24.79, MSE = 1590.82
                # 'subway_in_subway_out_calendar_input_dim2_stack_with_attn_d32_h4_ff128_l1': {'dataset_names': ['subway_in','calendar','subway_out'],
                #                                         'input_embedding_dim': 12,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #                                         'loss_function_type':'HuberLoss',
                #                                         'optimizer': 'adamw',
                #                                         'batch_size': 128,
                #                                         'epochs':500,
                #                                         'adaptive_embedding_dim': 32,
                #                                         'input_embedding_dim': 24,
                #                                         'tod_embedding_dim': 6,
                #                                         'dow_embedding_dim': 6,
                #                                         'feed_forward_dim': 256,
                                                        
                #                                         'num_heads': 4,
                #                                         'num_layers': 3,

                #                                         'use_mixed_proj': True,
                #                                         'freq': '15min',
                #                                         'H':6,
                #                                         'D':1,
                #                                         'W':0,

                #                                         'lr': 0.001,
                #                                         'weight_decay':  0.0015,
                #                                         'dropout': 0.2,
                #                                         'torch_scheduler_milestone': 20,
                #                                         'torch_scheduler_gamma':0.9925,
                #                                         'torch_scheduler_type': 'warmup',
                #                                         'torch_scheduler_lr_start_factor': 0.3,
                #                                         'standardize': True,
                #                                         'minmaxnorm': False,
                #                                         'calendar_types':['dayofweek', 'timeofday'],
                #                                         'contextual_kwargs' : {'subway_out': {'need_global_attn':True, 
                #                                                                             'stacked_contextual': True,
                #                                                                             'vision_model_name' : None,
                #                                                                             'use_only_for_common_dates': False, 
                #                                                                             'stack_consistent_datasets' : False, # True,
                #                                                                             'attn_kwargs': {'latent_dim': 1,
                #                                                                                             'dim_feedforward' : 128,
                #                                                                                             'num_heads':  4,
                #                                                                                             'dim_model':  32,
                #                                                                                             'keep_topk': False,
                #                                                                                             },
                #                                                                       },  
                #                                                         },
                #             },  

                # # All Steps RMSE = 39.94, MAE = 22.51, MAPE = 24.82, MSE = 1597.82
                # 'subway_in_subway_out_calendar_input_dim2_stack_with_attn_d64_h4_ff128_l1': {'dataset_names': ['subway_in','calendar','subway_out'],
                #                                         'input_embedding_dim': 12,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #                                         'loss_function_type':'HuberLoss',
                #                                         'optimizer': 'adamw',
                #                                         'batch_size': 128,
                #                                         'epochs':500,
                #                                         'adaptive_embedding_dim': 32,
                #                                         'input_embedding_dim': 24,
                #                                         'tod_embedding_dim': 6,
                #                                         'dow_embedding_dim': 6,
                #                                         'feed_forward_dim': 256,
                                                        
                #                                         'num_heads': 4,
                #                                         'num_layers': 3,

                #                                         'use_mixed_proj': True,
                #                                         'freq': '15min',
                #                                         'H':6,
                #                                         'D':1,
                #                                         'W':0,

                #                                         'lr': 0.001,
                #                                         'weight_decay':  0.0015,
                #                                         'dropout': 0.2,
                #                                         'torch_scheduler_milestone': 20,
                #                                         'torch_scheduler_gamma':0.9925,
                #                                         'torch_scheduler_type': 'warmup',
                #                                         'torch_scheduler_lr_start_factor': 0.3,
                #                                         'standardize': True,
                #                                         'minmaxnorm': False,
                #                                         'calendar_types':['dayofweek', 'timeofday'],
                #                                         'contextual_kwargs' : {'subway_out': {'need_global_attn':True, 
                #                                                                             'stacked_contextual': True,
                #                                                                             'vision_model_name' : None,
                #                                                                             'use_only_for_common_dates': False, 
                #                                                                             'stack_consistent_datasets' : False, # True,
                #                                                                             'attn_kwargs': {'latent_dim': 1,
                #                                                                                             'dim_feedforward' : 128,
                #                                                                                             'num_heads':  4,
                #                                                                                             'dim_model':  64,
                #                                                                                             'keep_topk': False,
                #                                                                                             },
                #                                                                       },  
                #                                                         },
                #             },      


                # # All Steps RMSE = 39.81, MAE = 22.42, MAPE = 25.28, MSE = 1587.32
                # 'subway_in_subway_out_calendar_input_dim2_stack_with_attn_d48_h3_ff64_l1': {'dataset_names': ['subway_in','calendar','subway_out'],
                #                                         'input_embedding_dim': 12,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #                                         'loss_function_type':'HuberLoss',
                #                                         'optimizer': 'adamw',
                #                                         'batch_size': 128,
                #                                         'epochs':500,
                #                                         'adaptive_embedding_dim': 32,
                #                                         'input_embedding_dim': 24,
                #                                         'tod_embedding_dim': 6,
                #                                         'dow_embedding_dim': 6,
                #                                         'feed_forward_dim': 256,
                                                        
                #                                         'num_heads': 4,
                #                                         'num_layers': 3,

                #                                         'use_mixed_proj': True,
                #                                         'freq': '15min',
                #                                         'H':6,
                #                                         'D':1,
                #                                         'W':0,

                #                                         'lr': 0.001,
                #                                         'weight_decay':  0.0015,
                #                                         'dropout': 0.2,
                #                                         'torch_scheduler_milestone': 20,
                #                                         'torch_scheduler_gamma':0.9925,
                #                                         'torch_scheduler_type': 'warmup',
                #                                         'torch_scheduler_lr_start_factor': 0.3,
                #                                         'standardize': True,
                #                                         'minmaxnorm': False,
                #                                         'calendar_types':['dayofweek', 'timeofday'],
                #                                         'contextual_kwargs' : {'subway_out': {'need_global_attn':True, 
                #                                                                             'stacked_contextual': True,
                #                                                                             'vision_model_name' : None,
                #                                                                             'use_only_for_common_dates': False, 
                #                                                                             'stack_consistent_datasets' : False, # True,
                #                                                                             'attn_kwargs': {'latent_dim': 1,
                #                                                                                             'dim_feedforward' : 64,
                #                                                                                             'num_heads':  3,
                #                                                                                             'dim_model':  48,
                #                                                                                             'keep_topk': False,
                #                                                                                             },
                #                                                                       },  
                #                                                         },
                #             },       




    #   #All Steps RMSE = 39.69, MAE = 22.47, MAPE = 25.27, MSE = 1576.47
    #   'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering02': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
    #                                             'loss_function_type':'HuberLoss',
    #                                             'optimizer': 'adamw',
    #                                             'batch_size': 128,
    #                                             'epochs':500,
    #                                             'adaptive_embedding_dim': 32,
    #                                             'tod_embedding_dim': 6,
    #                                             'dow_embedding_dim': 6,
    #                                             'feed_forward_dim': 256,
    #                                             'input_embedding_dim': 24,
    #                                             'num_heads': 4,
    #                                             'num_layers': 3,
    #                                             'lr': 0.001,
    #                                             'weight_decay':  0.0015,
    #                                             'dropout': 0.2,
    #                                             'torch_scheduler_milestone': 20,
    #                                             'torch_scheduler_gamma':0.9925,
    #                                             'torch_scheduler_type': 'warmup',
    #                                             'torch_scheduler_lr_start_factor': 0.3,

    #                                             'use_mixed_proj': True,
    #                                             'freq': '15min',
    #                                             'H':6,
    #                                             'D':1,
    #                                             'W':0,

    #                                             'standardize': True,
    #                                             'minmaxnorm': False,
    #                                             'calendar_types':['dayofweek', 'timeofday'],

    #                                             'unormalize_loss' : True,
    #                                             'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
    #                                                                                     'stacked_contextual': True,
    #                                                                                     'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
    #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
    #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
    #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
    #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
    #                                                                                     'vision_model_name' : None,
    #                                                                                     'epsilon_clustering': 0.2,
    #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
    #                                                                                     'attn_kwargs': {'latent_dim' : 2 ,
    #                                                                                                     'dim_feedforward' : 64,
    #                                                                                                     'num_heads' : 2 ,
    #                                                                                                     'dim_model' : 12,
    #                                                                                                     'keep_topk': False}  
    #                                                                                     #'H' : ,
    #                                                                                     #'D': ,
    #                                                                                     #'W': , 
    #                                                                         },
    #                                                                 },  
    #                                             'denoising_names':['netmob_POIs'],
    #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
    #                                             'denoising_modes':["train","valid","test"],             # par défaut
    #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
    #                 },

            #     # All Steps RMSE = 39.90, MAE = 22.55, MAPE = 24.91, MSE = 1593.66
            #    'calendar_Google_Maps_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering02': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
            #                                     'loss_function_type':'HuberLoss',
            #                                     'optimizer': 'adamw',
            #                                     'batch_size': 128,
            #                                     'epochs':500,
            #                                     'adaptive_embedding_dim': 32,
            #                                     'tod_embedding_dim': 6,
            #                                     'dow_embedding_dim': 6,
            #                                     'feed_forward_dim': 256,
            #                                     'input_embedding_dim': 24,
            #                                     'num_heads': 4,
            #                                     'num_layers': 3,
            #                                     'lr': 0.001,
            #                                     'weight_decay':  0.0015,
            #                                     'dropout': 0.2,
            #                                     'torch_scheduler_milestone': 20,
            #                                     'torch_scheduler_gamma':0.9925,
            #                                     'torch_scheduler_type': 'warmup',
            #                                     'torch_scheduler_lr_start_factor': 0.3,

            #                                     'use_mixed_proj': True,
            #                                     'freq': '15min',
            #                                     'H':6,
            #                                     'D':1,
            #                                     'W':0,

            #                                     'standardize': True,
            #                                     'minmaxnorm': False,
            #                                     'calendar_types':['dayofweek', 'timeofday'],

            #                                     'unormalize_loss' : True,
            #                                     'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
            #                                                                             'stacked_contextual': True,
            #                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
            #                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
            #                                                                             'NetMob_expanded' : '', # '' # '_expanded'
            #                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
            #                                                                             'vision_model_name' : None,
            #                                                                             'epsilon_clustering': 0.2,
            #                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
            #                                                                             'attn_kwargs': {'latent_dim' : 2 ,
            #                                                                                             'dim_feedforward' : 64,
            #                                                                                             'num_heads' : 2 ,
            #                                                                                             'dim_model' : 12,
            #                                                                                             'keep_topk': False}  
            #                                                                             #'H' : ,
            #                                                                             #'D': ,
            #                                                                             #'W': , 
            #                                                                 },
            #                                                         },  
            #                                     'denoising_names':['netmob_POIs'],
            #                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                     'denoising_modes':["train","valid","test"],             # par défaut
            #                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
            #         },

            #     # All Steps RMSE = 39.98, MAE = 22.61, MAPE = 25.38, MSE = 1601.93
            #    'calendar_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
            #                                     'loss_function_type':'HuberLoss',
            #                                     'optimizer': 'adamw',
            #                                     'batch_size': 128,
            #                                     'epochs':500,
            #                                     'adaptive_embedding_dim': 32,
            #                                     'tod_embedding_dim': 6,
            #                                     'dow_embedding_dim': 6,
            #                                     'feed_forward_dim': 256,
            #                                     'input_embedding_dim': 24,
            #                                     'num_heads': 4,
            #                                     'num_layers': 3,
            #                                     'lr': 0.001,
            #                                     'weight_decay':  0.0015,
            #                                     'dropout': 0.2,
            #                                     'torch_scheduler_milestone': 20,
            #                                     'torch_scheduler_gamma':0.9925,
            #                                     'torch_scheduler_type': 'warmup',
            #                                     'torch_scheduler_lr_start_factor': 0.3,

            #                                     'use_mixed_proj': True,
            #                                     'freq': '15min',
            #                                     'H':6,
            #                                     'D':1,
            #                                     'W':0,

            #                                     'standardize': True,
            #                                     'minmaxnorm': False,
            #                                     'calendar_types':['dayofweek', 'timeofday'],

            #                                     'unormalize_loss' : True,
            #                                     'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
            #                                                                             'stacked_contextual': True,
            #                                                                             'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
            #                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
            #                                                                             'NetMob_expanded' : '', # '' # '_expanded'
            #                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
            #                                                                             'vision_model_name' : None,
            #                                                                             'epsilon_clustering': 0.1,
            #                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
            #                                                                             'attn_kwargs': {'latent_dim' : 2 ,
            #                                                                                             'dim_feedforward' : 64,
            #                                                                                             'num_heads' : 2 ,
            #                                                                                             'dim_model' : 12,
            #                                                                                             'keep_topk': False}  
            #                                                                             #'H' : ,
            #                                                                             #'D': ,
            #                                                                             #'W': , 
            #                                                                 },
            #                                                         },  
            #                                     'denoising_names':['netmob_POIs'],
            #                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                     'denoising_modes':["train","valid","test"],             # par défaut
            #                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
            #         },


























                # MSE: 1613.87 # MAE: 22.76
                # 'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                #                                 'loss_function_type':'HuberLoss',
                #                                 'optimizer': 'adamw',
                #                                 'batch_size': 128,
                #                                 'epochs':500,
                #                                 'adaptive_embedding_dim': 32,
                #                                 'tod_embedding_dim': 6,
                #                                 'dow_embedding_dim': 6,
                #                                 'feed_forward_dim': 256,
                #                                 'input_embedding_dim': 24,
                #                                 'num_heads': 4,
                #                                 'num_layers': 3,
                #                                 'lr': 0.001,
                #                                 'weight_decay':  0.0015,
                #                                 'dropout': 0.2,
                #                                 'torch_scheduler_milestone': 20,
                #                                 'torch_scheduler_gamma':0.9925,
                #                                 'torch_scheduler_type': 'warmup',
                #                                 'torch_scheduler_lr_start_factor': 0.3,

                #                                 'use_mixed_proj': True,
                #                                 'freq': '15min',
                #                                 'H':6,
                #                                 'D':1,
                #                                 'W':0,

                #                                 'standardize': True,
                #                                 'minmaxnorm': False,
                #                                 'calendar_types':['dayofweek', 'timeofday'],

                #                                 'unormalize_loss' : True,
                #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                #                                                                         'stacked_contextual': True,
                #                                                                         'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                #                                                                         'vision_model_name' : None,
                #                                                                         'epsilon_clustering': 0.1,
                #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                #                                                                         'attn_kwargs': {'latent_dim' : 2 ,
                #                                                                                         'dim_feedforward' : 64,
                #                                                                                         'num_heads' : 2 ,
                #                                                                                         'dim_model' : 12,
                #                                                                                         'keep_topk': False}  
                #                                                                         #'H' : ,
                #                                                                         #'D': ,
                #                                                                         #'W': , 
                #                                                             },
                #                                                     },  
                #                                 'denoising_names':['netmob_POIs'],
                #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                #                                 'denoising_modes':["train","valid","test"],             # par défaut
                #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                #     },
######==========================================================================
#        BEST MODEL OBTENU 
#######========================================================================



    #   # All Steps RMSE = 40.57, MAE = 22.87, MAPE = 25.16, MSE = 1648.74
    #   'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering025': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
    #                                             'loss_function_type':'HuberLoss',
    #                                             'optimizer': 'adamw',
    #                                             'batch_size': 128,
    #                                             'epochs':500,
    #                                             'adaptive_embedding_dim': 32,
    #                                             'tod_embedding_dim': 6,
    #                                             'dow_embedding_dim': 6,
    #                                             'feed_forward_dim': 256,
    #                                             'input_embedding_dim': 24,
    #                                             'num_heads': 4,
    #                                             'num_layers': 3,
    #                                             'lr': 0.001,
    #                                             'weight_decay':  0.0015,
    #                                             'dropout': 0.2,
    #                                             'torch_scheduler_milestone': 20,
    #                                             'torch_scheduler_gamma':0.9925,
    #                                             'torch_scheduler_type': 'warmup',
    #                                             'torch_scheduler_lr_start_factor': 0.3,

    #                                             'use_mixed_proj': True,
    #                                             'freq': '15min',
    #                                             'H':6,
    #                                             'D':1,
    #                                             'W':0,

    #                                             'standardize': True,
    #                                             'minmaxnorm': False,
    #                                             'calendar_types':['dayofweek', 'timeofday'],

    #                                             'unormalize_loss' : True,
    #                                             'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
    #                                                                                     'stacked_contextual': True,
    #                                                                                     'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
    #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
    #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
    #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
    #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
    #                                                                                     'vision_model_name' : None,
    #                                                                                     'epsilon_clustering': 0.25,
    #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
    #                                                                                     'attn_kwargs': {'latent_dim' : 2 ,
    #                                                                                                     'dim_feedforward' : 64,
    #                                                                                                     'num_heads' : 2 ,
    #                                                                                                     'dim_model' : 12,
    #                                                                                                     'keep_topk': False}  
    #                                                                                     #'H' : ,
    #                                                                                     #'D': ,
    #                                                                                     #'W': , 
    #                                                                         },
    #                                                                 },  
    #                                             'denoising_names':['netmob_POIs'],
    #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
    #                                             'denoising_modes':["train","valid","test"],             # par défaut
    #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
    #                 },


    #     # All Steps RMSE = 41.13, MAE = 23.18, MAPE = 25.26, MSE = 1693.97
    #   'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering03': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
    #                                             'loss_function_type':'HuberLoss',
    #                                             'optimizer': 'adamw',
    #                                             'batch_size': 128,
    #                                             'epochs':500,
    #                                             'adaptive_embedding_dim': 32,
    #                                             'tod_embedding_dim': 6,
    #                                             'dow_embedding_dim': 6,
    #                                             'feed_forward_dim': 256,
    #                                             'input_embedding_dim': 24,
    #                                             'num_heads': 4,
    #                                             'num_layers': 3,
    #                                             'lr': 0.001,
    #                                             'weight_decay':  0.0015,
    #                                             'dropout': 0.2,
    #                                             'torch_scheduler_milestone': 20,
    #                                             'torch_scheduler_gamma':0.9925,
    #                                             'torch_scheduler_type': 'warmup',
    #                                             'torch_scheduler_lr_start_factor': 0.3,

    #                                             'use_mixed_proj': True,
    #                                             'freq': '15min',
    #                                             'H':6,
    #                                             'D':1,
    #                                             'W':0,

    #                                             'standardize': True,
    #                                             'minmaxnorm': False,
    #                                             'calendar_types':['dayofweek', 'timeofday'],

    #                                             'unormalize_loss' : True,
    #                                             'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
    #                                                                                     'stacked_contextual': True,
    #                                                                                     'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
    #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
    #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
    #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
    #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
    #                                                                                     'vision_model_name' : None,
    #                                                                                     'epsilon_clustering': 0.3,
    #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
    #                                                                                     'attn_kwargs': {'latent_dim' : 2 ,
    #                                                                                                     'dim_feedforward' : 64,
    #                                                                                                     'num_heads' : 2 ,
    #                                                                                                     'dim_model' : 12,
    #                                                                                                     'keep_topk': False}  
    #                                                                                     #'H' : ,
    #                                                                                     #'D': ,
    #                                                                                     #'W': , 
    #                                                                         },
    #                                                                 },  
    #                                             'denoising_names':['netmob_POIs'],
    #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
    #                                             'denoising_modes':["train","valid","test"],             # par défaut
    #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
    #                 },
       

    #     # All Steps RMSE = 41.59, MAE = 23.54, MAPE = 26.72, MSE = 1735.65
    #     'calendar_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering015': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
    #                                             'loss_function_type':'HuberLoss',
    #                                             'optimizer': 'adamw',
    #                                             'batch_size': 128,
    #                                             'epochs':500,
    #                                             'adaptive_embedding_dim': 32,
    #                                             'tod_embedding_dim': 6,
    #                                             'dow_embedding_dim': 6,
    #                                             'feed_forward_dim': 256,
    #                                             'input_embedding_dim': 24,
    #                                             'num_heads': 4,
    #                                             'num_layers': 3,
    #                                             'lr': 0.001,
    #                                             'weight_decay':  0.0015,
    #                                             'dropout': 0.2,
    #                                             'torch_scheduler_milestone': 20,
    #                                             'torch_scheduler_gamma':0.9925,
    #                                             'torch_scheduler_type': 'warmup',
    #                                             'torch_scheduler_lr_start_factor': 0.3,

    #                                             'use_mixed_proj': True,
    #                                             'freq': '15min',
    #                                             'H':6,
    #                                             'D':1,
    #                                             'W':0,

    #                                             'standardize': True,
    #                                             'minmaxnorm': False,
    #                                             'calendar_types':['dayofweek', 'timeofday'],

    #                                             'unormalize_loss' : True,
    #                                             'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
    #                                                                                     'stacked_contextual': True,
    #                                                                                     'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
    #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
    #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
    #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
    #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
    #                                                                                     'vision_model_name' : None,
    #                                                                                     'epsilon_clustering': 0.15,
    #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
    #                                                                                     'attn_kwargs': {'latent_dim' : 2 ,
    #                                                                                                     'dim_feedforward' : 64,
    #                                                                                                     'num_heads' : 2 ,
    #                                                                                                     'dim_model' : 12,
    #                                                                                                     'keep_topk': False}  
    #                                                                                     #'H' : ,
    #                                                                                     #'D': ,
    #                                                                                     #'W': , 
    #                                                                         },
    #                                                                 },  
    #                                             'denoising_names':['netmob_POIs'],
    #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
    #                                             'denoising_modes':["train","valid","test"],             # par défaut
    #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
    #                 },


                
    #             # All Steps RMSE = 40.11, MAE = 22.68, MAPE = 25.81, MSE = 1610.26
    #            'calendar_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering02': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
    #                                             'loss_function_type':'HuberLoss',
    #                                             'optimizer': 'adamw',
    #                                             'batch_size': 128,
    #                                             'epochs':500,
    #                                             'adaptive_embedding_dim': 32,
    #                                             'tod_embedding_dim': 6,
    #                                             'dow_embedding_dim': 6,
    #                                             'feed_forward_dim': 256,
    #                                             'input_embedding_dim': 24,
    #                                             'num_heads': 4,
    #                                             'num_layers': 3,
    #                                             'lr': 0.001,
    #                                             'weight_decay':  0.0015,
    #                                             'dropout': 0.2,
    #                                             'torch_scheduler_milestone': 20,
    #                                             'torch_scheduler_gamma':0.9925,
    #                                             'torch_scheduler_type': 'warmup',
    #                                             'torch_scheduler_lr_start_factor': 0.3,

    #                                             'use_mixed_proj': True,
    #                                             'freq': '15min',
    #                                             'H':6,
    #                                             'D':1,
    #                                             'W':0,

    #                                             'standardize': True,
    #                                             'minmaxnorm': False,
    #                                             'calendar_types':['dayofweek', 'timeofday'],

    #                                             'unormalize_loss' : True,
    #                                             'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
    #                                                                                     'stacked_contextual': True,
    #                                                                                     'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
    #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
    #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
    #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
    #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
    #                                                                                     'vision_model_name' : None,
    #                                                                                     'epsilon_clustering': 0.2,
    #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
    #                                                                                     'attn_kwargs': {'latent_dim' : 2 ,
    #                                                                                                     'dim_feedforward' : 64,
    #                                                                                                     'num_heads' : 2 ,
    #                                                                                                     'dim_model' : 12,
    #                                                                                                     'keep_topk': False}  
    #                                                                                     #'H' : ,
    #                                                                                     #'D': ,
    #                                                                                     #'W': , 
    #                                                                         },
    #                                                                 },  
    #                                             'denoising_names':['netmob_POIs'],
    #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
    #                                             'denoising_modes':["train","valid","test"],             # par défaut
    #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
    #                 },

    #      # All Steps RMSE = 40.29, MAE = 22.74, MAPE = 25.34, MSE = 1626.31
    #     'calendar_Google_Maps_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering015': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
    #                                             'loss_function_type':'HuberLoss',
    #                                             'optimizer': 'adamw',
    #                                             'batch_size': 128,
    #                                             'epochs':500,
    #                                             'adaptive_embedding_dim': 32,
    #                                             'tod_embedding_dim': 6,
    #                                             'dow_embedding_dim': 6,
    #                                             'feed_forward_dim': 256,
    #                                             'input_embedding_dim': 24,
    #                                             'num_heads': 4,
    #                                             'num_layers': 3,
    #                                             'lr': 0.001,
    #                                             'weight_decay':  0.0015,
    #                                             'dropout': 0.2,
    #                                             'torch_scheduler_milestone': 20,
    #                                             'torch_scheduler_gamma':0.9925,
    #                                             'torch_scheduler_type': 'warmup',
    #                                             'torch_scheduler_lr_start_factor': 0.3,

    #                                             'use_mixed_proj': True,
    #                                             'freq': '15min',
    #                                             'H':6,
    #                                             'D':1,
    #                                             'W':0,

    #                                             'standardize': True,
    #                                             'minmaxnorm': False,
    #                                             'calendar_types':['dayofweek', 'timeofday'],

    #                                             'unormalize_loss' : True,
    #                                             'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
    #                                                                                     'stacked_contextual': True,
    #                                                                                     'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
    #                                                                                     'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
    #                                                                                     'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
    #                                                                                     'NetMob_expanded' : '', # '' # '_expanded'
    #                                                                                     'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
    #                                                                                     'vision_model_name' : None,
    #                                                                                     'epsilon_clustering': 0.15,
    #                                                                                     'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
    #                                                                                     'attn_kwargs': {'latent_dim' : 2 ,
    #                                                                                                     'dim_feedforward' : 64,
    #                                                                                                     'num_heads' : 2 ,
    #                                                                                                     'dim_model' : 12,
    #                                                                                                     'keep_topk': False}  
    #                                                                                     #'H' : ,
    #                                                                                     #'D': ,
    #                                                                                     #'W': , 
    #                                                                         },
    #                                                                 },  
    #                                             'denoising_names':['netmob_POIs'],
    #                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
    #                                             'denoising_modes':["train","valid","test"],             # par défaut
    #                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
    #                 },




            #    'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering005': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
            #                                     'loss_function_type':'HuberLoss',
            #                                     'optimizer': 'adamw',
            #                                     'batch_size': 128,
            #                                     'epochs':500,
            #                                     'adaptive_embedding_dim': 32,
            #                                     'tod_embedding_dim': 6,
            #                                     'dow_embedding_dim': 6,
            #                                     'feed_forward_dim': 256,
            #                                     'input_embedding_dim': 24,
            #                                     'num_heads': 4,
            #                                     'num_layers': 3,
            #                                     'lr': 0.001,
            #                                     'weight_decay':  0.0015,
            #                                     'dropout': 0.2,
            #                                     'torch_scheduler_milestone': 20,
            #                                     'torch_scheduler_gamma':0.9925,
            #                                     'torch_scheduler_type': 'warmup',
            #                                     'torch_scheduler_lr_start_factor': 0.3,

            #                                     'use_mixed_proj': True,
            #                                     'freq': '15min',
            #                                     'H':6,
            #                                     'D':1,
            #                                     'W':0,

            #                                     'standardize': True,
            #                                     'minmaxnorm': False,
            #                                     'calendar_types':['dayofweek', 'timeofday'],

            #                                     'unormalize_loss' : True,
            #                                     'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
            #                                                                             'stacked_contextual': True,
            #                                                                             'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
            #                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
            #                                                                             'NetMob_expanded' : '', # '' # '_expanded'
            #                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
            #                                                                             'vision_model_name' : None,
            #                                                                             'epsilon_clustering': 0.05,
            #                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
            #                                                                             'attn_kwargs': {'latent_dim' : 2 ,
            #                                                                                             'dim_feedforward' : 64,
            #                                                                                             'num_heads' : 2 ,
            #                                                                                             'dim_model' : 12,
            #                                                                                             'keep_topk': False}  
            #                                                                             #'H' : ,
            #                                                                             #'D': ,
            #                                                                             #'W': , 
            #                                                                 },
            #                                                         },  
            #                                     'denoising_names':['netmob_POIs'],
            #                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                     'denoising_modes':["train","valid","test"],             # par défaut
            #                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
            #         },
            #      # All Steps RMSE = 40.29, MAE = 22.98, MAPE = 25.70, MSE = 1626.38
            #      'calendar_Google_Maps_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering01': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
            #                                     'loss_function_type':'HuberLoss',
            #                                     'optimizer': 'adamw',
            #                                     'batch_size': 128,
            #                                     'epochs':500,
            #                                     'adaptive_embedding_dim': 32,
            #                                     'tod_embedding_dim': 6,
            #                                     'dow_embedding_dim': 6,
            #                                     'feed_forward_dim': 256,
            #                                     'input_embedding_dim': 24,
            #                                     'num_heads': 4,
            #                                     'num_layers': 3,
            #                                     'lr': 0.001,
            #                                     'weight_decay':  0.0015,
            #                                     'dropout': 0.2,
            #                                     'torch_scheduler_milestone': 20,
            #                                     'torch_scheduler_gamma':0.9925,
            #                                     'torch_scheduler_type': 'warmup',
            #                                     'torch_scheduler_lr_start_factor': 0.3,

            #                                     'use_mixed_proj': True,
            #                                     'freq': '15min',
            #                                     'H':6,
            #                                     'D':1,
            #                                     'W':0,

            #                                     'standardize': True,
            #                                     'minmaxnorm': False,
            #                                     'calendar_types':['dayofweek', 'timeofday'],

            #                                     'unormalize_loss' : True,
            #                                     'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
            #                                                                             'stacked_contextual': True,
            #                                                                             'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
            #                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
            #                                                                             'NetMob_expanded' : '', # '' # '_expanded'
            #                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
            #                                                                             'vision_model_name' : None,
            #                                                                             'epsilon_clustering': 0.1,
            #                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
            #                                                                             'attn_kwargs': {'latent_dim' : 2 ,
            #                                                                                             'dim_feedforward' : 64,
            #                                                                                             'num_heads' : 2 ,
            #                                                                                             'dim_model' : 12,
            #                                                                                             'keep_topk': False}  
            #                                                                             #'H' : ,
            #                                                                             #'D': ,
            #                                                                             #'W': , 
            #                                                                 },
            #                                                         },  
            #                                     'denoising_names':['netmob_POIs'],
            #                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                     'denoising_modes':["train","valid","test"],             # par défaut
            #                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
            #         },
            # 'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering0125': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
            #                                     'loss_function_type':'HuberLoss',
            #                                     'optimizer': 'adamw',
            #                                     'batch_size': 128,
            #                                     'epochs':500,
            #                                     'adaptive_embedding_dim': 32,
            #                                     'tod_embedding_dim': 6,
            #                                     'dow_embedding_dim': 6,
            #                                     'feed_forward_dim': 256,
            #                                     'input_embedding_dim': 24,
            #                                     'num_heads': 4,
            #                                     'num_layers': 3,
            #                                     'lr': 0.001,
            #                                     'weight_decay':  0.0015,
            #                                     'dropout': 0.2,
            #                                     'torch_scheduler_milestone': 20,
            #                                     'torch_scheduler_gamma':0.9925,
            #                                     'torch_scheduler_type': 'warmup',
            #                                     'torch_scheduler_lr_start_factor': 0.3,

            #                                     'use_mixed_proj': True,
            #                                     'freq': '15min',
            #                                     'H':6,
            #                                     'D':1,
            #                                     'W':0,

            #                                     'standardize': True,
            #                                     'minmaxnorm': False,
            #                                     'calendar_types':['dayofweek', 'timeofday'],

            #                                     'unormalize_loss' : True,
            #                                     'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
            #                                                                             'stacked_contextual': True,
            #                                                                             'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
            #                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
            #                                                                             'NetMob_expanded' : '', # '' # '_expanded'
            #                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
            #                                                                             'vision_model_name' : None,
            #                                                                             'epsilon_clustering': 0.125,
            #                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
            #                                                                             'attn_kwargs': {'latent_dim' : 2 ,
            #                                                                                             'dim_feedforward' : 64,
            #                                                                                             'num_heads' : 2 ,
            #                                                                                             'dim_model' : 12,
            #                                                                                             'keep_topk': False}  
            #                                                                             #'H' : ,
            #                                                                             #'D': ,
            #                                                                             #'W': , 
            #                                                                 },
            #                                                         },  
            #                                     'denoising_names':['netmob_POIs'],
            #                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                     'denoising_modes':["train","valid","test"],             # par défaut
            #                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
            #         },

            # 'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_clustering015': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
            #                                     'loss_function_type':'HuberLoss',
            #                                     'optimizer': 'adamw',
            #                                     'batch_size': 128,
            #                                     'epochs':500,
            #                                     'adaptive_embedding_dim': 32,
            #                                     'tod_embedding_dim': 6,
            #                                     'dow_embedding_dim': 6,
            #                                     'feed_forward_dim': 256,
            #                                     'input_embedding_dim': 24,
            #                                     'num_heads': 4,
            #                                     'num_layers': 3,
            #                                     'lr': 0.001,
            #                                     'weight_decay':  0.0015,
            #                                     'dropout': 0.2,
            #                                     'torch_scheduler_milestone': 20,
            #                                     'torch_scheduler_gamma':0.9925,
            #                                     'torch_scheduler_type': 'warmup',
            #                                     'torch_scheduler_lr_start_factor': 0.3,

            #                                     'use_mixed_proj': True,
            #                                     'freq': '15min',
            #                                     'H':6,
            #                                     'D':1,
            #                                     'W':0,

            #                                     'standardize': True,
            #                                     'minmaxnorm': False,
            #                                     'calendar_types':['dayofweek', 'timeofday'],

            #                                     'unormalize_loss' : True,
            #                                     'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
            #                                                                             'stacked_contextual': True,
            #                                                                             'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
            #                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
            #                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
            #                                                                             'NetMob_expanded' : '', # '' # '_expanded'
            #                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
            #                                                                             'vision_model_name' : None,
            #                                                                             'epsilon_clustering': 0.15,
            #                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
            #                                                                             'attn_kwargs': {'latent_dim' : 2 ,
            #                                                                                             'dim_feedforward' : 64,
            #                                                                                             'num_heads' : 2 ,
            #                                                                                             'dim_model' : 12,
            #                                                                                             'keep_topk': False}  
            #                                                                             #'H' : ,
            #                                                                             #'D': ,
            #                                                                             #'W': , 
            #                                                                 },
            #                                                         },  
            #                                     'denoising_names':['netmob_POIs'],
            #                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            #                                     'denoising_modes':["train","valid","test"],             # par défaut
            #                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
            #         },


                # 'calendar_input_dim24_calib_prop90': {'dataset_names': ['subway_in','calendar'],
                #                                         'input_embedding_dim': 12,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #                                         'loss_function_type':'HuberLoss',
                #                                         'optimizer': 'adamw',
                #                                         'batch_size': 128,
                #                                         'epochs':500,
                #                                         'adaptive_embedding_dim': 32,
                #                                         'input_embedding_dim': 24,
                #                                         'tod_embedding_dim': 6,
                #                                         'dow_embedding_dim': 6,
                #                                         'feed_forward_dim': 256,
                                                        
                #                                         'num_heads': 4,
                #                                         'num_layers': 3,

                #                                         'use_mixed_proj': True,
                #                                         'freq': '15min',
                #                                         'H':6,
                #                                         'D':1,
                #                                         'W':0,
                #                                         'calib_prop':0.9,

                #                                         'lr': 0.001,
                #                                         'weight_decay':  0.0015,
                #                                         'dropout': 0.2,
                #                                         'torch_scheduler_milestone': 20,
                #                                         'torch_scheduler_gamma':0.9925,
                #                                         'torch_scheduler_type': 'warmup',
                #                                         'torch_scheduler_lr_start_factor': 0.3,
                #                                         'standardize': True,
                #                                         'minmaxnorm': False,
                #                                         'calendar_types':['dayofweek', 'timeofday']
                #             },


                # 'calendar_input_dim24_calib_prop50': {'dataset_names': ['subway_in','calendar'],
                #                                         'input_embedding_dim': 12,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #                                         'loss_function_type':'HuberLoss',
                #                                         'optimizer': 'adamw',
                #                                         'batch_size': 128,
                #                                         'epochs':500,
                #                                         'adaptive_embedding_dim': 32,
                #                                         'input_embedding_dim': 24,
                #                                         'tod_embedding_dim': 6,
                #                                         'dow_embedding_dim': 6,
                #                                         'feed_forward_dim': 256,
                                                        
                #                                         'num_heads': 4,
                #                                         'num_layers': 3,

                #                                         'use_mixed_proj': True,
                #                                         'freq': '15min',
                #                                         'H':6,
                #                                         'D':1,
                #                                         'W':0,
                #                                         'calib_prop':0.5,

                #                                         'lr': 0.001,
                #                                         'weight_decay':  0.0015,
                #                                         'dropout': 0.2,
                #                                         'torch_scheduler_milestone': 20,
                #                                         'torch_scheduler_gamma':0.9925,
                #                                         'torch_scheduler_type': 'warmup',
                #                                         'torch_scheduler_lr_start_factor': 0.3,
                #                                         'standardize': True,
                #                                         'minmaxnorm': False,
                #                                         'calendar_types':['dayofweek', 'timeofday']
                #             },

                # 'calendar_input_dim24_calib_prop20': {'dataset_names': ['subway_in','calendar'],
                #                                         'input_embedding_dim': 12,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #                                         'loss_function_type':'HuberLoss',
                #                                         'optimizer': 'adamw',
                #                                         'batch_size': 128,
                #                                         'epochs':500,
                #                                         'adaptive_embedding_dim': 32,
                #                                         'input_embedding_dim': 24,
                #                                         'tod_embedding_dim': 6,
                #                                         'dow_embedding_dim': 6,
                #                                         'feed_forward_dim': 256,
                                                        
                #                                         'num_heads': 4,
                #                                         'num_layers': 3,

                #                                         'use_mixed_proj': True,
                #                                         'freq': '15min',
                #                                         'H':6,
                #                                         'D':1,
                #                                         'W':0,
                #                                         'calib_prop':0.2,

                #                                         'lr': 0.001,
                #                                         'weight_decay':  0.0015,
                #                                         'dropout': 0.2,
                #                                         'torch_scheduler_milestone': 20,
                #                                         'torch_scheduler_gamma':0.9925,
                #                                         'torch_scheduler_type': 'warmup',
                #                                         'torch_scheduler_lr_start_factor': 0.3,
                #                                         'standardize': True,
                #                                         'minmaxnorm': False,
                #                                         'calendar_types':['dayofweek', 'timeofday']
                #             },


                # 'calendar_input_dim24_calib_prop10': {'dataset_names': ['subway_in','calendar'],
                #                                         'input_embedding_dim': 12,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #                                         'loss_function_type':'HuberLoss',
                #                                         'optimizer': 'adamw',
                #                                         'batch_size': 128,
                #                                         'epochs':500,
                #                                         'adaptive_embedding_dim': 32,
                #                                         'input_embedding_dim': 24,
                #                                         'tod_embedding_dim': 6,
                #                                         'dow_embedding_dim': 6,
                #                                         'feed_forward_dim': 256,
                                                        
                #                                         'num_heads': 4,
                #                                         'num_layers': 3,

                #                                         'use_mixed_proj': True,
                #                                         'freq': '15min',
                #                                         'H':6,
                #                                         'D':1,
                #                                         'W':0,
                #                                         'calib_prop':0.1,

                #                                         'lr': 0.001,
                #                                         'weight_decay':  0.0015,
                #                                         'dropout': 0.2,
                #                                         'torch_scheduler_milestone': 20,
                #                                         'torch_scheduler_gamma':0.9925,
                #                                         'torch_scheduler_type': 'warmup',
                #                                         'torch_scheduler_lr_start_factor': 0.3,
                #                                         'standardize': True,
                #                                         'minmaxnorm': False,
                #                                         'calendar_types':['dayofweek', 'timeofday']
                #             },





#                 # 'baseline': {  'dataset_names': ['subway_in','calendar'],
#                 #                 'loss_function_type':'MSE',
#                 #                 'input_embedding_dim': 12,
#                 #                 'tod_embedding_dim': 0,
#                 #                 'dow_embedding_dim': 0,
#                 #                 'spatial_embedding_dim': 0,
#                 #                 'adaptive_embedding_dim': 20,
#                 #                 'num_heads': 4,
#                 #                 'num_layers': 3,
#                 #                 'feed_forward_dim': 128,
#                 #                 'use_mixed_proj': True,
#                 #                 'weight_decay': 0.0014059383370107,
#                 #                 'batch_size': 128,
#                 #                 'lr': 0.0018507902690237,
#                 #                 'dropout': 0.15,
#                 #                 'epochs': 300,
#                 #                 'scheduler': True,
#                 #                 'torch_scheduler_milestone': 50,
#                 #                 'torch_scheduler_gamma': 0.9935177454064168,
#                 #                 'torch_scheduler_lr_start_factor': 0.6140839981178771,
#                 #                 'torch_scheduler_type': 'warmup',

#                 #                 'standardize': False,
#                 #                 'minmaxnorm': True,
#                 #             },

#                 #     'baseline2': {'dataset_names': ['subway_in','calendar'],
#                 #                             'loss_function_type':'MSE',
#                 #                                 'lr': 0.00105, # 5e-5,# 4e-4,
#                 #                                 'weight_decay': 0.0188896655584368, # 0.05,
#                 #                                 'dropout': 0.271795372610271, # 0.15,

#                 #                                 'scheduler': True,  # None
#                 #                                 'torch_scheduler_type': 'warmup',
#                 #                                 'torch_scheduler_milestone': 28.0, #5,
#                 #                                 'torch_scheduler_gamma': 0.9958348861339396, # 0.997,
#                 #                                 'torch_scheduler_lr_start_factor': 0.8809942312067847, # 1,

#                 #                                 'input_embedding_dim': 12,
#                 #                                 'adaptive_embedding_dim':20,
#                 #                                 'tod_embedding_dim':0,
#                 #                                 'dow_embedding_dim':0,

#                 #                                 'standardize': False,
#                 #                                 'minmaxnorm': True,
#                 #                                 },
                                                
#                     'MSE': {'dataset_names': ['subway_in','calendar'],
#                                 'loss_function_type':'MSE',
#                                     'lr': 1e-3, # 5e-5,# 4e-4,
#                                     'weight_decay': 0.066, # 0.05,
#                                     'dropout': 0.15, # 0.15,

#                                     'scheduler': True,  # None
#                                     'torch_scheduler_type': 'warmup',
#                                     'torch_scheduler_milestone': 40, #5,
#                                     'torch_scheduler_gamma': 0.995, # 0.997,
#                                     'torch_scheduler_lr_start_factor': 0.9, # 1,

#                                     'input_embedding_dim': 12,
#                                     'adaptive_embedding_dim':20,
#                                     'tod_embedding_dim':0,
#                                     'dow_embedding_dim':0,

#                                     'standardize': False,
#                                     'minmaxnorm': True,
#                                     },
#                 'HuberLoss': {'dataset_names': ['subway_in','calendar'],
#                                 'loss_function_type':'HuberLoss',
#                                     'lr': 1e-3, # 5e-5,# 4e-4,
#                                     'weight_decay': 0.066, # 0.05,
#                                     'dropout': 0.15, # 0.15,

#                                     'scheduler': True,  # None
#                                     'torch_scheduler_type': 'warmup',
#                                     'torch_scheduler_milestone': 40, #5,
#                                     'torch_scheduler_gamma': 0.995, # 0.997,
#                                     'torch_scheduler_lr_start_factor': 0.9, # 1,

#                                     'input_embedding_dim': 12,
#                                     'adaptive_embedding_dim':20,
#                                     'tod_embedding_dim':0,
#                                     'dow_embedding_dim':0,

#                                     'standardize': False,
#                                     'minmaxnorm': True,
#                                     },
#                 'HuberLoss_standardize': {'dataset_names': ['subway_in','calendar'],
#                                 'loss_function_type':'HuberLoss',
#                                     'lr': 1e-3, # 5e-5,# 4e-4,
#                                     'weight_decay': 0.066, # 0.05,
#                                     'dropout': 0.15, # 0.15,
                                    
#                                     'scheduler': True,  # None
#                                     'torch_scheduler_type': 'warmup',
#                                     'torch_scheduler_milestone': 40, #5,
#                                     'torch_scheduler_gamma': 0.995, # 0.997,
#                                     'torch_scheduler_lr_start_factor': 0.9, # 1,

#                                     'input_embedding_dim': 12,
#                                     'adaptive_embedding_dim':20,
#                                     'tod_embedding_dim':0,
#                                     'dow_embedding_dim':0,

#                                     'standardize': True,
#                                     'minmaxnorm': False,
#                                     },
        # 'Huber_MultiStepLr_standardize': {
        #         'lr': 0.001,
        #         'weight_decay':  0.0015,
        #         'dropout': 0.2,
        #                     },

        # 'Huber_MultiStepLr_Tuned_standardize': {
        #         'lr': 0.00105, # 5e-5,# 4e-4,
        #         'weight_decay': 0.0188896655584368, # 0.05,
        #         'dropout': 0.271795372610271, # 0.15,
        #                     },   

        # 'Huber_MultiStepLr_Tuned_MinMax': {
        #         'lr': 0.00105, # 5e-5,# 4e-4,
        #         'weight_decay': 0.0188896655584368, # 0.05,
        #         'dropout': 0.271795372610271, # 0.15,

        #         'standardize': False,
        #         'minmaxnorm': True, 
        #                     },   


        # 'Huber_WarmUp_standardize': {
        #         'lr': 0.001,
        #         'weight_decay':  0.0015,
        #         'dropout': 0.2,
        #         'torch_scheduler_milestone': 20,
        #         'torch_scheduler_gamma':0.9925,
        #         'torch_scheduler_type': 'warmup',
        #         'torch_scheduler_lr_start_factor': 0.3,

        #         'standardize': True,
        #         'minmaxnorm': False, 
        #                     }, 

        # 'Huber_WarmUp_MinMax': {
        #         'lr': 0.001,
        #         'weight_decay':  0.0015,
        #         'dropout': 0.2,
        #         'torch_scheduler_milestone': 20,
        #         'torch_scheduler_gamma':0.9925,
        #         'torch_scheduler_type': 'warmup',
        #         'torch_scheduler_lr_start_factor': 0.3,

        #         'standardize': False,
        #         'minmaxnorm': True, 
        #                     },  

        # 'Huber_WarmUp_Tuned_MinMax': {
        #         'lr': 0.00105, # 5e-5,# 4e-4,
        #         'weight_decay': 0.0188896655584368, # 0.05,
        #         'dropout': 0.271795372610271, # 0.15,
        #         'torch_scheduler_milestone': 20,
        #         'torch_scheduler_gamma':0.9925,
        #         'torch_scheduler_type': 'warmup',
        #         'torch_scheduler_lr_start_factor': 0.3,

        #         'standardize': False,
        #         'minmaxnorm': True, 
        #                     },
                              
        # 'Huber_WarmUp_standardize_adp32_emb12': {
        #         'adaptive_embedding_dim': 32,
        #         'input_embedding_dim': 12,
        #         'tod_embedding_dim': 12,
        #         'dow_embedding_dim': 12,
        #         'feed_forward_dim': 256,

        #         'lr': 0.001,
        #         'weight_decay':  0.0015,
        #         'dropout': 0.2,
        #         'torch_scheduler_milestone': 20,
        #         'torch_scheduler_gamma':0.9925,
        #         'torch_scheduler_type': 'warmup',
        #         'torch_scheduler_lr_start_factor': 0.3,

        #         'standardize': True,
        #         'minmaxnorm': False, 
        #                     }, 

        # 'Huber_WarmUp_standardize_adp32_emb12_ff128': {
        #         'adaptive_embedding_dim': 32,
        #         'input_embedding_dim': 12,
        #         'tod_embedding_dim': 12,
        #         'dow_embedding_dim': 12,
        #         'feed_forward_dim': 128,

        #         'lr': 0.001,
        #         'weight_decay':  0.0015,
        #         'dropout': 0.2,
        #         'torch_scheduler_milestone': 20,
        #         'torch_scheduler_gamma':0.9925,
        #         'torch_scheduler_type': 'warmup',
        #         'torch_scheduler_lr_start_factor': 0.3,

        #         'standardize': True,
        #         'minmaxnorm': False, 
        #                     }, 

        # 'Huber_WarmUp_standardize_adp32_emb12_6_6': {
        #         'adaptive_embedding_dim': 32,
        #         'input_embedding_dim': 12,
        #         'tod_embedding_dim': 6,
        #         'dow_embedding_dim': 6,
        #         'feed_forward_dim': 256,

        #         'lr': 0.001,
        #         'weight_decay':  0.0015,
        #         'dropout': 0.2,
        #         'torch_scheduler_milestone': 20,
        #         'torch_scheduler_gamma':0.9925,
        #         'torch_scheduler_type': 'warmup',
        #         'torch_scheduler_lr_start_factor': 0.3,

        #         'standardize': True,
        #         'minmaxnorm': False, 
        #                     },



        # 'Huber_WarmUp_standardize_adp32_emb12_6_6_ff128': {
        #         'adaptive_embedding_dim': 32,
        #         'input_embedding_dim': 12,
        #         'tod_embedding_dim': 6,
        #         'dow_embedding_dim': 6,
        #         'feed_forward_dim': 128,

        #         'lr': 0.001,
        #         'weight_decay':  0.0015,
        #         'dropout': 0.2,
        #         'torch_scheduler_milestone': 20,
        #         'torch_scheduler_gamma':0.9925,
        #         'torch_scheduler_type': 'warmup',
        #         'torch_scheduler_lr_start_factor': 0.3,

        #         'standardize': True,
        #         'minmaxnorm': False, 
        #                     }, 
    #  'no_calendar_tod_dow_0': {'dataset_names': ['subway_in','calendar'],
    #                            'denoising_names': [] ,
    #                             'tod_embedding_dim': 0,
    #                             'dow_embedding_dim': 0,
    #                            'NetMob_selected_apps'  :  [], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
    #                            },
    
    #                 'calendar': {'dataset_names': ['subway_in','calendar'],
    #                                         'denoising_names': [] ,
    #                                         'NetMob_selected_apps'  :  [], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
    #                                         },
                        #     'calendar_Deezer_eps300_input_dim24': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'denoising_names':[],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                         'stacked_contextual': True,
                        #                                                         'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                        #                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                             'NetMob_selected_tags' : ['station_epsilon300'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                             'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                             'NetMob_only_epsilon': True, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                             'vision_model_name' : None,
                        #                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                             },
                        #                                                     },  
                        #     },

                        #    'calendar_Google_Map_eps300_input_dim24': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                         'denoising_names':[],
                        #                                        'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                         'stacked_contextual': True,
                        #                                                         'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
                        #                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                             'NetMob_selected_tags' : ['station_epsilon300'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                             'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                             'NetMob_only_epsilon': True, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                             'vision_model_name' : None,
                        #                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                             },
                        #                                                     },    
                        #     },
                        #    'calendar_Web_Weather_eps300_input_dim24': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                         'denoising_names':[],
                        #                                          'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                         'stacked_contextual': True,
                        #                                                         'NetMob_selected_apps' : ['Web_Weather'], # Google_Maps # Web_Weather # Deezer
                        #                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                             'NetMob_selected_tags' : ['station_epsilon300'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                             'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                             'NetMob_only_epsilon': True, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                             'vision_model_name' : None,
                        #                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                             },
                        #                                                     },    
                        #     },

                        #    'calendar_Web_Weather_Deezer_eps300_input_dim24': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                         'denoising_names':[],
                        #                                          'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                         'stacked_contextual': True,
                        #                                                         'NetMob_selected_apps' : ['Web_Weather','Deezer'], # Google_Maps # Web_Weather # Deezer
                        #                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                             'NetMob_selected_tags' : ['station_epsilon300'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                             'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                             'NetMob_only_epsilon': True, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                             'vision_model_name' : None,
                        #                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                             },
                        #                                                     },    
                        #     },

                        #         'calendar_Google_Maps_Deezer_eps300_input_dim24': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                         'denoising_names':[],
                        #                                          'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                         'stacked_contextual': True,
                        #                                                         'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # Web_Weather # Deezer
                        #                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                             'NetMob_selected_tags' : ['station_epsilon300'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                             'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                             'NetMob_only_epsilon': True, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                             'vision_model_name' : None,
                        #                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                             },
                        #                                                     },   
                        #     },                            


                        # 'calendar_input_dim24': {'dataset_names': ['subway_in','calendar'],
                        #                                 'denoising_names':[],
                        #                                 'contextual_kwargs' : {},
                        #                                 'input_embedding_dim': 12*2,
                                                                            
                        #    },  


                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h3_ldim2': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'denoising_names':[],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 2 ,
                        #                                                                                         'dim_feedforward' : 64,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 48,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #     },
                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h1_ldim4': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'denoising_names':[],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                         'dim_feedforward' : 64,
                        #                                                                                         'num_heads' : 1 ,
                        #                                                                                         'dim_model' : 48,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #     },

                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h1_ldim2': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'denoising_names':[],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 2 ,
                        #                                                                                         'dim_feedforward' : 64,
                        #                                                                                         'num_heads' : 1 ,
                        #                                                                                         'dim_model' : 48,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #     },
                    #     'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h1_ldim1': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                    #                                     'denoising_names':[],
                    #                                     'input_embedding_dim': 12*2,
                    #                                     'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                    #                                                                             'stacked_contextual': True,
                    #                                                                             'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                    #                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                    #                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                    #                                                                             'NetMob_expanded' : '', # '' # '_expanded'
                    #                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                    #                                                                             'vision_model_name' : None,
                    #                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                    #                                                                             'attn_kwargs': {'latent_dim' : 1 ,
                    #                                                                                             'dim_feedforward' : 64,
                    #                                                                                             'num_heads' : 1 ,
                    #                                                                                             'dim_model' : 48,}  
                    #                                                                             #'H' : ,
                    #                                                                             #'D': ,
                    #                                                                             #'W': , 
                    #                                                                 },
                    #                                                         },  
                    #         },

                    #     'calendar_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                    #                                     'denoising_names':[],
                    #                                     'input_embedding_dim': 12*2,
                    #                                     'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                    #                                                                             'stacked_contextual': True,
                    #                                                                             'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                    #                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                    #                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                    #                                                                             'NetMob_expanded' : '', # '' # '_expanded'
                    #                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                    #                                                                             'vision_model_name' : None,
                    #                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                    #                                                                             'attn_kwargs': {'latent_dim' : 2 ,
                    #                                                                                             'dim_feedforward' : 64,
                    #                                                                                             'num_heads' : 2 ,
                    #                                                                                             'dim_model' : 64,}  
                    #                                                                             #'H' : ,
                    #                                                                             #'D': ,
                    #                                                                             #'W': , 
                    #                                                                 },
                    #                                                         },  
                    #         },

                    #   'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h1_ldim1_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                    #                                     'denoising_names':[],
                    #                                     'input_embedding_dim': 12*2,
                    #                                     'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                    #                                                                             'stacked_contextual': True,
                    #                                                                             'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                    #                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                    #                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                    #                                                                             'NetMob_expanded' : '', # '' # '_expanded'
                    #                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                    #                                                                             'vision_model_name' : None,
                    #                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                    #                                                                             'attn_kwargs': {'latent_dim' : 1 ,
                    #                                                                                             'dim_feedforward' : 64,
                    #                                                                                             'num_heads' : 1 ,
                    #                                                                                             'dim_model' : 48,}  
                    #                                                                             #'H' : ,
                    #                                                                             #'D': ,
                    #                                                                             #'W': , 
                    #                                                                 },
                    #                                     'denoising_names':['netmob_POIs'],
                    #                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                    #                                     'denoising_modes':["train","valid","test"],             # par défaut
                    #                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                    #                                                         },  
                    #         },

                    #     'calendar_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                    #                                     'denoising_names':[],
                    #                                     'input_embedding_dim': 12*2,
                    #                                     'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                    #                                                                             'stacked_contextual': True,
                    #                                                                             'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                    #                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                    #                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                    #                                                                             'NetMob_expanded' : '', # '' # '_expanded'
                    #                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                    #                                                                             'vision_model_name' : None,
                    #                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                    #                                                                             'attn_kwargs': {'latent_dim' : 2 ,
                    #                                                                                             'dim_feedforward' : 64,
                    #                                                                                             'num_heads' : 2 ,
                    #                                                                                             'dim_model' : 64,}  
                    #                                                                             #'H' : ,
                    #                                                                             #'D': ,
                    #                                                                             #'W': , 
                    #                                                                 },
                    #                                     'denoising_names':['netmob_POIs'],
                    #                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                    #                                     'denoising_modes':["train","valid","test"],             # par défaut
                    #                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                    #                                                         },  
                    #         },
                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h3_ldim1': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'denoising_names':[],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 1 ,
                        #                                                                                         'dim_feedforward' : 64,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 48,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #     },

                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h2_ldim4': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'denoising_names':[],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                         'dim_feedforward' : 64,
                        #                                                                                         'num_heads' : 2 ,
                        #                                                                                         'dim_model' : 48,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #     },

                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h3_ldim4': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'denoising_names':[],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                         'dim_feedforward' : 64,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 48,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #     },

                        #    'calendar_Google_Maps_IRIS_input_dim24_attn_dim48_ff64_h3_ldim4': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                         'denoising_names':[],
                        #                                        'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                         'dim_feedforward' : 64,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 48,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },   
                        #                               },

                        #  'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim48_ff64_h3_ldim4': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                         'denoising_names':[],
                        #                                          'input_embedding_dim': 12*2,
                        #                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                                 'stacked_contextual': True,
                        #                                                                                 'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
                        #                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                                 'vision_model_name' : None,
                        #                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                                 'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                                 'dim_feedforward' : 64,
                        #                                                                                                 'num_heads' : 3 ,
                        #                                                                                                 'dim_model' : 48,}  
                        #                                                                                 #'H' : ,
                        #                                                                                 #'D': ,
                        #                                                                                 #'W': , 
                        #                                                                     },
                        #                                                             },   
                        #     },  
        


    #   'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_HP3': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
    #                                                             'denoising_names':[],
    #                                                             'lr':  0.00175035,
    #                                                             'weight_decay':0.00203832,
    #                                                             'dropout': 0.1   ,
    #                                                             'torch_scheduler_milestone': 5,
    #                                                             'torch_scheduler_gamma':   0.98754,
    #                                                             'torch_scheduler_type': 'warmup',
    #                                                             'torch_scheduler_lr_start_factor': 0.552248 ,
    #                                                             'epochs':500,

    #                                                 'loss_function_type':'HuberLoss',
    #                                                 'optimizer': 'adamw',
    #                                                 'adaptive_embedding_dim': 32,
    #                                                 'tod_embedding_dim': 6,
    #                                                 'dow_embedding_dim': 6,
    #                                                 'feed_forward_dim': 256,

    #                                                 'num_heads': 4,
    #                                                 'num_layers': 3,

    #                                                 'use_mixed_proj': True,
    #                                                 'freq': '15min',
    #                                                 'H':6,
    #                                                 'D':1,
    #                                                 'W':0,

    #                                                 'input_embedding_dim': 24,
    #                                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
    #                                                                                             'stacked_contextual': True,
    #                                                                                             'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
    #                                                                                             'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
    #                                                                                             'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
    #                                                                                             'NetMob_expanded' : '', # '' # '_expanded'
    #                                                                                             'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
    #                                                                                             'vision_model_name' : None,
    #                                                                                             'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
    #                                                                                             'attn_kwargs': {'latent_dim' : 2 ,
    #                                                                                                             'dim_feedforward' : 64,
    #                                                                                                             'num_heads' : 2 ,
    #                                                                                                             'dim_model' : 64
    #                                                                                                             }  
    #                                                                                             #'H' : ,
    #                                                                                             #'D': ,
    #                                                                                             #'W': , 
    #                                                                                 },
    #                                                                         },  
    #                                                     'denoising_names':['netmob_POIs'],
    #                                                     'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
    #                                                     'denoising_modes':["train","valid","test"],             # par défaut
    #                                                     'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
    #                       },

# 0.000831384      0.00340202         0.15                        1                 0.982239                 0.359437      500 
# 0.0017567        0.00819653         0.05                       10                 0.982889                 0.897474      500
# 0.00149806       0.00146675         0.05                        1                 0.985742                 0.382134      500 
# 0.00169937       0.00644466         0.05                       10                 0.981606                 0.734226      500
# 0.00109771       0.00399033         0.15                        1                 0.982501                 0.654805      500 
# 0.000888038      0.00895351         0.05                        1                 0.980851                 0.364287      500
# 0.00126734       0.00727456         0.05                       10                 0.983451                 0.370938      500 
# 0.00180552       0.00421896         0                          10                 0.985636                 0.863735      500 
# 0.00253259       0.00125364         0.1                        10                 0.981973                 0.44482       500
# 0.00197368       0.00412883         0                          10                 0.984939                 0.700371      500
# 0.00102584       0.000565536        0.05                        1                 0.983491                 0.456887      500

# 0.00204685       0.00227278         0                           1                 0.984016                 0.316197      500 
# 0.00206827       0.00567128         0.1                        10                 0.983595                 0.446556      500 

# 0.00176897       0.00119972         0.1                         5                 0.989365                 0.553424      500
# 0.0014543        0.00163937         0.1                         5                 0.982812                 0.584919      500 
#   0.00250536       0.00192661         0                           5                 0.985426                 0.628773      500 

#   0.00124175       0.00962615         0.1                         5                 0.986409                 0.346245      500 
#   0.00175035       0.00203832         0.1                         5                 0.98754                  0.552248      500 
#   0.00178889       0.00812294         0.05                        5                 0.990794                 0.457083      500
#   0.00144143       0.00685895         0.1                         5                 0.988808                 0.499934      500



                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h3_ldim4_exp_smooth_07': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'denoising_names':[],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                         'dim_feedforward' : 64,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 48,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.7}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },


                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h3_ldim4_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                         'dim_feedforward' : 64,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 48,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },

                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h3_ldim4_exp_smooth_09': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                         'dim_feedforward' : 64,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 48,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.9}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },

                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h3_ldim1_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 1 ,
                        #                                                                                         'dim_feedforward' : 64,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 48,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },

                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h3_ldim1_exp_smooth_09': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 1,
                        #                                                                                         'dim_feedforward' : 64,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 48,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.9}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },

                        

                        # 'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim48_ff64_h3_ldim4_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                         'dim_feedforward' : 64,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 48,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },

                        # 'calendar_Deezer_Google_Maps_IRIS_input_dim24_attn_dim48_ff64_h3_ldim4_exp_smooth_09': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                         'dim_feedforward' : 64,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 48,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.9}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },



                #         'calendar_input_dim12_NormalizedLoss': {'dataset_names': ['subway_in','calendar'],
                #                                         'input_embedding_dim': 12,
                #                                         'unormalize_loss' : False,
                #                                         'contextual_kwargs' : {},  
                #             },
                #          'calendar_input_dim24_NormalizedLoss': {'dataset_names': ['subway_in','calendar'],
                #                                         'input_embedding_dim': 24,
                #                                         'unormalize_loss' : False,
                #                                         'contextual_kwargs' : {},  
                #             },
                #          'calendar_input_dim32_NormalizedLoss': {'dataset_names': ['subway_in','calendar'],
                #                                         'input_embedding_dim': 32,
                #                                         'unormalize_loss' : False,
                #                                         'contextual_kwargs' : {},  
                #             },
                #          'calendar_input_dim24_lr_HP_NormalizedLoss': {'dataset_names': ['subway_in','calendar'],
                #                                         'input_embedding_dim': 24,
                #                                         'unormalize_loss' : False,
                #                                         'contextual_kwargs' : {},  
                #                                         'weight_decay': 0.0019180662821482, 
                #                                         'lr': 0.0009241967812616,            
                #                                         'dropout': 0.05,       
                #                                         'epochs': 500,
                #                                         'scheduler': True,
                #                                         'torch_scheduler_milestone': 1,
                #                                         'torch_scheduler_gamma': 0.9816450698678711,   
                #                                         'torch_scheduler_lr_start_factor': 0.4123017434871985,   
                #             },

                #         'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_HP_tuning_NormalizedLoss': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                #                                         'input_embedding_dim': 24,
                #                                         'unormalize_loss' : False,
                #                                         'weight_decay': 0.0019180662821482, 
                #                                         'lr': 0.0009241967812616,            
                #                                         'dropout': 0.05,       
                #                                         'epochs': 500,
                #                                         'scheduler': True,
                #                                         'torch_scheduler_milestone': 1,
                #                                         'torch_scheduler_gamma': 0.9816450698678711,   
                #                                         'torch_scheduler_lr_start_factor': 0.4123017434871985,    
                #                                         'torch_scheduler_type': 'warmup',    
                #                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                #                                                                                 'stacked_contextual': True,
                #                                                                                 'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                #                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                #                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                #                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
                #                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                #                                                                                 'vision_model_name' : None,
                #                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                #                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
                #                                                                                                 'dim_feedforward' : 64,
                #                                                                                                 'num_heads' : 2 ,
                #                                                                                                 'dim_model' : 64,}  
                #                                                                                 #'H' : ,
                #                                                                                 #'D': ,
                #                                                                                 #'W': , 
                #                                                                     },
                #                                                             },  
                #                                         'denoising_names':['netmob_POIs'],
                #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                #                                         'denoising_modes':["train","valid","test"],             # par défaut
                #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                #             },
                # 'calendar_Google_Maps_Deezer_IRIS_input_dim32_attn_dim64_ff64_h2_ldim2_exp_smooth_08_HP_tuning_NormalizedLoss': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                #                                         'input_embedding_dim': 32,
                #                                         'unormalize_loss' : False,
                #                                         'weight_decay': 0.0019180662821482, 
                #                                         'lr': 0.0009241967812616,            
                #                                         'dropout': 0.05,       
                #                                         'epochs': 500,
                #                                         'scheduler': True,
                #                                         'torch_scheduler_milestone': 1,
                #                                         'torch_scheduler_gamma': 0.9816450698678711,   
                #                                         'torch_scheduler_lr_start_factor': 0.4123017434871985,    
                #                                         'torch_scheduler_type': 'warmup',    
                #                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                #                                                                                 'stacked_contextual': True,
                #                                                                                 'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                #                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                #                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                #                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
                #                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                #                                                                                 'vision_model_name' : None,
                #                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                #                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
                #                                                                                                 'dim_feedforward' : 64,
                #                                                                                                 'num_heads' : 2 ,
                #                                                                                                 'dim_model' : 64,}  
                #                                                                                 #'H' : ,
                #                                                                                 #'D': ,
                #                                                                                 #'W': , 
                #                                                                     },
                #                                                             },  
                #                                         'denoising_names':['netmob_POIs'],
                #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                #                                         'denoising_modes':["train","valid","test"],             # par défaut
                #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                #             },
                #         'calendar_Google_Maps_Deezer_IRIS_input_dim12_attn_dim64_ff64_h2_ldim2_exp_smooth_08_NormalizedLoss': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                #                                         'input_embedding_dim': 24,
                #                                         'unormalize_loss' : False,
                #                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                #                                                                                 'stacked_contextual': True,
                #                                                                                 'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                #                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                #                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                #                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
                #                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                #                                                                                 'vision_model_name' : None,
                #                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                #                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
                #                                                                                                 'dim_feedforward' : 64,
                #                                                                                                 'num_heads' : 2 ,
                #                                                                                                 'dim_model' : 12,}  
                #                                                                                 #'H' : ,
                #                                                                                 #'D': ,
                #                                                                                 #'W': , 
                #                                                                     },
                #                                                             },  
                #                                         'denoising_names':['netmob_POIs'],
                #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                #                                         'denoising_modes':["train","valid","test"],             # par défaut
                #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                #             },

                #         'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_NormalizedLoss': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                #                                         'input_embedding_dim': 24,
                #                                         'unormalize_loss' : False,
                #                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                #                                                                                 'stacked_contextual': True,
                #                                                                                 'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                #                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                #                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                #                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
                #                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                #                                                                                 'vision_model_name' : None,
                #                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                #                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
                #                                                                                                 'dim_feedforward' : 64,
                #                                                                                                 'num_heads' : 2 ,
                #                                                                                                 'dim_model' : 64,}  
                #                                                                                 #'H' : ,
                #                                                                                 #'D': ,
                #                                                                                 #'W': , 
                #                                                                     },
                #                                                             },  
                #                                         'denoising_names':['netmob_POIs'],
                #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                #                                         'denoising_modes':["train","valid","test"],             # par défaut
                #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                #             },
                #         'calendar_Google_Maps_Deezer_IRIS_input_dim32_attn_dim64_ff64_h2_ldim2_exp_smooth_08_NormalizedLoss': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                #                                         'input_embedding_dim': 32,
                #                                         'unormalize_loss' : False,
                #                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                #                                                                                 'stacked_contextual': True,
                #                                                                                 'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                #                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                #                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                #                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
                #                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                #                                                                                 'vision_model_name' : None,
                #                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                #                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
                #                                                                                                 'dim_feedforward' : 64,
                #                                                                                                 'num_heads' : 2 ,
                #                                                                                                 'dim_model' : 64,}  
                #                                                                                 #'H' : ,
                #                                                                                 #'D': ,
                #                                                                                 #'W': , 
                #                                                                     },
                #                                                             },  
                #                                         'denoising_names':['netmob_POIs'],
                #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                #                                         'denoising_modes':["train","valid","test"],             # par défaut
                #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                #             },


















                #         'calendar_input_dim12': {'dataset_names': ['subway_in','calendar'],
                #                                         'input_embedding_dim': 12,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #             },
                #          'calendar_input_dim24': {'dataset_names': ['subway_in','calendar'],
                #                                         'input_embedding_dim': 24,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #             },
                #          'calendar_input_dim32': {'dataset_names': ['subway_in','calendar'],
                #                                         'input_embedding_dim': 32,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #             },
                #          'calendar_input_dim24_lr_HP': {'dataset_names': ['subway_in','calendar'],
                #                                         'input_embedding_dim': 24,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {},  
                #                                         'weight_decay': 0.0019180662821482, 
                #                                         'lr': 0.0009241967812616,            
                #                                         'dropout': 0.05,       
                #                                         'epochs': 500,
                #                                         'scheduler': True,
                #                                         'torch_scheduler_milestone': 1,
                #                                         'torch_scheduler_gamma': 0.9816450698678711,   
                #                                         'torch_scheduler_lr_start_factor': 0.4123017434871985,   
                #             },

                #         'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08_HP_tuning': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                #                                         'input_embedding_dim': 24,
                #                                         'unormalize_loss' : True,
                #                                         'weight_decay': 0.0019180662821482, 
                #                                         'lr': 0.0009241967812616,            
                #                                         'dropout': 0.05,       
                #                                         'epochs': 500,
                #                                         'scheduler': True,
                #                                         'torch_scheduler_milestone': 1,
                #                                         'torch_scheduler_gamma': 0.9816450698678711,   
                #                                         'torch_scheduler_lr_start_factor': 0.4123017434871985,    
                #                                         'torch_scheduler_type': 'warmup',    
                #                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                #                                                                                 'stacked_contextual': True,
                #                                                                                 'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                #                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                #                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                #                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
                #                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                #                                                                                 'vision_model_name' : None,
                #                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                #                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
                #                                                                                                 'dim_feedforward' : 64,
                #                                                                                                 'num_heads' : 2 ,
                #                                                                                                 'dim_model' : 64,}  
                #                                                                                 #'H' : ,
                #                                                                                 #'D': ,
                #                                                                                 #'W': , 
                #                                                                     },
                #                                                             },  
                #                                         'denoising_names':['netmob_POIs'],
                #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                #                                         'denoising_modes':["train","valid","test"],             # par défaut
                #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                #             },
                # 'calendar_Google_Maps_Deezer_IRIS_input_dim32_attn_dim64_ff64_h2_ldim2_exp_smooth_08_HP_tuning': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                #                                         'input_embedding_dim': 32,
                #                                         'unormalize_loss' : True,
                #                                         'weight_decay': 0.0019180662821482, 
                #                                         'lr': 0.0009241967812616,            
                #                                         'dropout': 0.05,       
                #                                         'epochs': 500,
                #                                         'scheduler': True,
                #                                         'torch_scheduler_milestone': 1,
                #                                         'torch_scheduler_gamma': 0.9816450698678711,   
                #                                         'torch_scheduler_lr_start_factor': 0.4123017434871985,    
                #                                         'torch_scheduler_type': 'warmup',    
                #                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                #                                                                                 'stacked_contextual': True,
                #                                                                                 'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                #                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                #                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                #                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
                #                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                #                                                                                 'vision_model_name' : None,
                #                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                #                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
                #                                                                                                 'dim_feedforward' : 64,
                #                                                                                                 'num_heads' : 2 ,
                #                                                                                                 'dim_model' : 64,}  
                #                                                                                 #'H' : ,
                #                                                                                 #'D': ,
                #                                                                                 #'W': , 
                #                                                                     },
                #                                                             },  
                #                                         'denoising_names':['netmob_POIs'],
                #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                #                                         'denoising_modes':["train","valid","test"],             # par défaut
                #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                #             },
                #         'calendar_Google_Maps_Deezer_IRIS_input_dim12_attn_dim64_ff64_h2_ldim2_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                #                                         'input_embedding_dim': 24,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                #                                                                                 'stacked_contextual': True,
                #                                                                                 'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                #                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                #                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                #                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
                #                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                #                                                                                 'vision_model_name' : None,
                #                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                #                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
                #                                                                                                 'dim_feedforward' : 64,
                #                                                                                                 'num_heads' : 2 ,
                #                                                                                                 'dim_model' : 12,}  
                #                                                                                 #'H' : ,
                #                                                                                 #'D': ,
                #                                                                                 #'W': , 
                #                                                                     },
                #                                                             },  
                #                                         'denoising_names':['netmob_POIs'],
                #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                #                                         'denoising_modes':["train","valid","test"],             # par défaut
                #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                #             },

                #         'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                #                                         'input_embedding_dim': 24,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                #                                                                                 'stacked_contextual': True,
                #                                                                                 'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                #                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                #                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                #                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
                #                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                #                                                                                 'vision_model_name' : None,
                #                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                #                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
                #                                                                                                 'dim_feedforward' : 64,
                #                                                                                                 'num_heads' : 2 ,
                #                                                                                                 'dim_model' : 64,}  
                #                                                                                 #'H' : ,
                #                                                                                 #'D': ,
                #                                                                                 #'W': , 
                #                                                                     },
                #                                                             },  
                #                                         'denoising_names':['netmob_POIs'],
                #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                #                                         'denoising_modes':["train","valid","test"],             # par défaut
                #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                #             },
                #         'calendar_Google_Maps_Deezer_IRIS_input_dim32_attn_dim64_ff64_h2_ldim2_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                #                                         'input_embedding_dim': 32,
                #                                         'unormalize_loss' : True,
                #                                         'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                #                                                                                 'stacked_contextual': True,
                #                                                                                 'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                #                                                                                 'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                #                                                                                 'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                #                                                                                 'NetMob_expanded' : '', # '' # '_expanded'
                #                                                                                 'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                #                                                                                 'vision_model_name' : None,
                #                                                                                 'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                #                                                                                 'attn_kwargs': {'latent_dim' : 2 ,
                #                                                                                                 'dim_feedforward' : 64,
                #                                                                                                 'num_heads' : 2 ,
                #                                                                                                 'dim_model' : 64,}  
                #                                                                                 #'H' : ,
                #                                                                                 #'D': ,
                #                                                                                 #'W': , 
                #                                                                     },
                #                                                             },  
                #                                         'denoising_names':['netmob_POIs'],
                #                                         'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                #                                         'denoising_modes':["train","valid","test"],             # par défaut
                #                                         'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                #             },




                        # 'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim96_ff128_h3_ldim4_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                         'dim_feedforward' : 128,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 96,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },
                        # 'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim96_ff128_h3_ldim2_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 2 ,
                        #                                                                                         'dim_feedforward' : 128,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 96,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },

                        # 'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim192_ff256_h3_ldim4_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                         'dim_feedforward' : 256,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 192,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },
                        # 'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim192_ff256_h3_ldim2_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 2 ,
                        #                                                                                         'dim_feedforward' : 256,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 192,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },


                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h3_ldim4_exp_smooth_05': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                         'dim_feedforward' : 64,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 48,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.5}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },


                        # 'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim192_ff256_h3_ldim8_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 8 ,
                        #                                                                                         'dim_feedforward' : 256,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 192,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },


                        # 'calendar_Google_Maps_IRIS_input_dim24_attn_dim192_ff256_h3_ldim8_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 8 ,
                        #                                                                                         'dim_feedforward' : 256,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 192,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },

                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim192_ff256_h3_ldim8_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 8 ,
                        #                                                                                         'dim_feedforward' : 256,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 192,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },


                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim192_ff256_h3_ldim4_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                         'dim_feedforward' : 256,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 192,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },

                        # 'calendar_Google_Maps_IRIS_input_dim24_attn_dim192_ff256_h3_ldim4_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                         'dim_feedforward' : 256,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 192,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },





                        #    'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim192_ff256_h3_ldim4_exp_smooth_08_e500': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'epochs':500,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                         'dim_feedforward' : 256,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 192,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },


                        # 'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim192_ff256_h3_ldim8_exp_smooth_08_e500': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'epochs':500,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 8 ,
                        #                                                                                         'dim_feedforward' : 256,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 192,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },


                        # 'calendar_Google_Maps_IRIS_input_dim24_attn_dim192_ff256_h3_ldim8_exp_smooth_08_e500': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                  'epochs':500,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 8 ,
                        #                                                                                         'dim_feedforward' : 256,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 192,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },

                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim192_ff256_h3_ldim8_exp_smooth_08_e500': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                  'epochs':500,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 8 ,
                        #                                                                                         'dim_feedforward' : 256,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 192,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },


                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim192_ff256_h3_ldim4_exp_smooth_08_e500': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'epochs':500,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                         'dim_feedforward' : 256,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 192,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },

                        # 'calendar_Google_Maps_IRIS_input_dim24_attn_dim192_ff256_h3_ldim4_exp_smooth_08_e500': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'epochs':500,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                        #                                                                         'stacked_contextual': True,
                        #                                                                         'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
                        #                                                                         'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                        #                                                                         'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                        #                                                                         'NetMob_expanded' : '', # '' # '_expanded'
                        #                                                                         'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                        #                                                                         'vision_model_name' : None,
                        #                                                                         'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                        #                                                                         'attn_kwargs': {'latent_dim' : 4 ,
                        #                                                                                         'dim_feedforward' : 256,
                        #                                                                                         'num_heads' : 3 ,
                        #                                                                                         'dim_model' : 192,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #                                 'denoising_names':['netmob_POIs'],
                        #                                 'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        #                                 'denoising_modes':["train","valid","test"],             # par défaut
                        #                                 'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        #     },
                       

    #                             'calendar_Google_Maps_Deezer_eps300_exp_smooth_07': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                                            
    #                                                             'denoising_names':['netmob_POIs'],
    #                                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
    #                                                             'denoising_modes':["train","valid","test"],             # par défaut
    #                                                             'denoiser_kwargs':{'exponential': {'alpha': 0.7}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                
    #                                                             'NetMob_selected_apps'  :  ['Google_Maps','Deezer'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
    #                                                             'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
    #                                                             'NetMob_selected_tags' : ['station_epsilon300'],
    #                                                             'NetMob_expanded'  : '' ,# '' # '_expanded'
    #                                                             'NetMob_only_epsilon':  True ,
    #                                                                         },
    #                             'calendar_Google_Maps_Deezer_eps300_exp_smooth_06': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                                            
    #                                                             'denoising_names':['netmob_POIs'],
    #                                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
    #                                                             'denoising_modes':["train","valid","test"],             # par défaut
    #                                                             'denoiser_kwargs':{'exponential': {'alpha': 0.6}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                
    #                                                             'NetMob_selected_apps'  :  ['Google_Maps','Deezer'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
    #                                                             'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
    #                                                             'NetMob_selected_tags' : ['station_epsilon300'],
    #                                                             'NetMob_expanded'  : '' ,# '' # '_expanded'
    #                                                             'NetMob_only_epsilon':  True ,
    #                                                                         },
    #                             'calendar_Google_Maps_Deezer_eps300_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                                            
    #                                                             'denoising_names':['netmob_POIs'],
    #                                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
    #                                                             'denoising_modes':["train","valid","test"],             # par défaut
    #                                                             'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                
    #                                                             'NetMob_selected_apps'  :  ['Google_Maps','Deezer'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
    #                                                             'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
    #                                                             'NetMob_selected_tags' : ['station_epsilon300'],
    #                                                             'NetMob_expanded'  : '' ,# '' # '_expanded'
    #                                                             'NetMob_only_epsilon':  True ,
    #                                                                         },
    #                             'calendar_Google_Maps_Deezer_eps300_exp_smooth_09': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                                            
    #                                                             'denoising_names':['netmob_POIs'],
    #                                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
    #                                                             'denoising_modes':["train","valid","test"],             # par défaut
    #                                                             'denoiser_kwargs':{'exponential': {'alpha': 0.9}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                
    #                                                             'NetMob_selected_apps'  :  ['Google_Maps','Deezer'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
    #                                                             'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
    #                                                             'NetMob_selected_tags' : ['station_epsilon300'],
    #                                                             'NetMob_expanded'  : '' ,# '' # '_expanded'
    #                                                             'NetMob_only_epsilon':  True ,
    #                                                                         },
    #                             'calendar_Google_Maps_Deezer_eps300_exp_smooth_05': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                                            
    #                                                             'denoising_names':['netmob_POIs'],
    #                                                             'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
    #                                                             'denoising_modes':["train","valid","test"],             # par défaut
    #                                                             'denoiser_kwargs':{'exponential': {'alpha': 0.5}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                
    #                                                             'NetMob_selected_apps'  :  ['Google_Maps','Deezer'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
    #                                                             'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
    #                                                             'NetMob_selected_tags' : ['station_epsilon300'],
    #                                                             'NetMob_expanded'  : '' ,# '' # '_expanded'
    #                                                             'NetMob_only_epsilon':  True ,
    #                                                                         },





    # 'adapt_32_emb_dim_12_ff256_h4l3_mse_loss': {
    #                     'loss_function_type':'MSE',
    #                     'optimizer': 'adamw',
    #                     'batch_size': 128,
    #                     'epochs':500,
    #                     'adaptive_embedding_dim': 32,
    #                     'input_embedding_dim': 12,
    #                     'tod_embedding_dim': 12,
    #                     'dow_embedding_dim': 12,
    #                     'feed_forward_dim': 256,
    #                     'num_heads': 4,
    #                     'num_layers': 3,
    #                     'freq': '15min',
    #                     'H':6,
    #                     'D':1,
    #                     'W':0,

    #                     'optimizer': 'adamw',

    #                     'batch_size': 128,
    #                     'lr': 0.001,
    #                     'weight_decay':  0.0015,
    #                     'dropout': 0.2,
    #                     'torch_scheduler_milestone': 20,
    #                     'torch_scheduler_gamma':0.9925,
    #                     'torch_scheduler_type': 'warmup',
    #                     'torch_scheduler_lr_start_factor': 0.3,

    #                     'standardize': True,
    #                     'minmaxnorm': False,
    #                     'calendar_types':['dayofweek', 'timeofday']
    #                                 },

    #           'adapt_32_emb_dim_12_ff256_h4l3_Huber_loss': {
    #                     'loss_function_type':'HuberLoss',
    #                     'optimizer': 'adamw',
    #                     'batch_size': 128,
    #                     'epochs':500,
    #                     'adaptive_embedding_dim': 32,
    #                     'input_embedding_dim': 12,
    #                     'tod_embedding_dim': 12,
    #                     'dow_embedding_dim': 12,
    #                     'feed_forward_dim': 256,
    #                     'num_heads': 4,
    #                     'num_layers': 3,
    #                     'freq': '15min',
    #                     'H':6,
    #                     'D':1,
    #                     'W':0,
    #                     'optimizer': 'adamw',

    #                     'batch_size': 128,
    #                     'lr': 0.001,
    #                     'weight_decay':  0.0015,
    #                     'dropout': 0.2,
    #                     'torch_scheduler_milestone': 20,
    #                     'torch_scheduler_gamma':0.9925,
    #                     'torch_scheduler_type': 'warmup',
    #                     'torch_scheduler_lr_start_factor': 0.3,

    #                      'standardize': True,
    #                      'minmaxnorm': False, 
    #                      'calendar_types':['dayofweek', 'timeofday']
    #                                 },

    #           'adapt_64_emb_dim_12_ff256_h4l3_Huber_loss': {
    #                     'loss_function_type':'HuberLoss',
    #                     'optimizer': 'adamw',
    #                     'batch_size': 128,
    #                     'epochs':500,
    #                     'adaptive_embedding_dim': 64,
    #                     'input_embedding_dim': 12,
    #                     'tod_embedding_dim': 12,
    #                     'dow_embedding_dim': 12,
    #                     'feed_forward_dim': 256,
    #                     'num_heads': 4,
    #                     'num_layers': 3,
    #                     'freq': '15min',
    #                     'H':6,
    #                     'D':1,
    #                     'W':0,
    #                     'optimizer': 'adamw',

    #                     'batch_size': 128,
    #                     'lr': 0.001,
    #                     'weight_decay':  0.0015,
    #                     'dropout': 0.2,
    #                     'torch_scheduler_milestone': 20,
    #                     'torch_scheduler_gamma':0.9925,
    #                     'torch_scheduler_type': 'warmup',
    #                     'torch_scheduler_lr_start_factor': 0.3,

    #                      'standardize': True,
    #                      'minmaxnorm': False, 
    #                      'calendar_types':['dayofweek', 'timeofday']
    #                                 },

                # 'adapt_64_emb_dim_24_ff256_h4l3_Huber_loss': {
                #         'loss_function_type':'HuberLoss',
                #         'optimizer': 'adamw',
                #         'batch_size': 128,
                #         'epochs':500,
                #         'adaptive_embedding_dim': 64,
                #         'input_embedding_dim': 24,
                #         'tod_embedding_dim': 24,
                #         'dow_embedding_dim': 24,
                #         'feed_forward_dim': 256,
                #         'num_heads': 4,
                #         'num_layers': 3,
                #         'freq': '15min',
                #         'H':6,
                #         'D':1,
                #         'W':0,
                #         'optimizer': 'adamw',

                #         'batch_size': 128,
                #         'lr': 0.001,
                #         'weight_decay':  0.0015,
                #         'dropout': 0.2,
                #         'torch_scheduler_milestone': 20,
                #         'torch_scheduler_gamma':0.9925,
                #         'torch_scheduler_type': 'warmup',
                #         'torch_scheduler_lr_start_factor': 0.3,

                #          'standardize': True,
                #          'minmaxnorm': False, 
                #          'calendar_types':['dayofweek', 'timeofday']
                #                     },

    #             'adapt_64_emb_dim_24_ff256_h4l3_Huber_loss_H6D0W0': {
    #                     'loss_function_type':'HuberLoss',
    #                     'optimizer': 'adamw',
    #                     'batch_size': 128,
    #                     'epochs':500,
    #                     'adaptive_embedding_dim': 64,
    #                     'input_embedding_dim': 24,
    #                     'tod_embedding_dim': 24,
    #                     'dow_embedding_dim': 24,
    #                     'feed_forward_dim': 256,
    #                     'num_heads': 4,
    #                     'num_layers': 3,
    #                     'freq': '15min',
    #                     'H':6,
    #                     'D':0,
    #                     'W':0,
    #                     'optimizer': 'adamw',

    #                     'batch_size': 128,
    #                     'lr': 0.001,
    #                     'weight_decay':  0.0015,
    #                     'dropout': 0.2,
    #                     'torch_scheduler_milestone': 20,
    #                     'torch_scheduler_gamma':0.9925,
    #                     'torch_scheduler_type': 'warmup',
    #                     'torch_scheduler_lr_start_factor': 0.3,

    #                      'standardize': True,
    #                      'minmaxnorm': False, 
    #                      'calendar_types':['dayofweek', 'timeofday']
    #                                 },
    #                         
    }

if len(constant_modif) > 0:
    modifications_bis = modifications.copy()
    modifications = {}
    for key, value in modifications_bis.items():
        modif_i = constant_modif.copy()
        modif_i.update(value)
        modifications[f"{constant_name}_{key}"] = modif_i