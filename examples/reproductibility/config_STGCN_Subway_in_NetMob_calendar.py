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

#  'subway_in_calendar_emb64_out64_Huber_MinMax_bis': {'target_data': 'subway_in',
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


######==========================================================================
#        BEST MODEL OBTENU 
#######========================================================================


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