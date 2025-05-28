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
constant_modif = {'optimizer': 'adamw',
                    'batch_size': 128,
                    'epochs':300,

                    'freq': '15min',
                    'H':6,
                    'D':1,
                    'W':0,
                        }

modifications = {'subway_in_MSE_MinMax': {'target_data': 'subway_in',
                                          'dataset_names': ['subway_in'],
                                    'loss_function_type':'MSE',
                                    'Kt': 2,
                                    'stblock_num': 2,
                                    'Ks': 2,
                                    'graph_conv_type': 'graph_conv',
                                    'gso_type': 'sym_renorm_adj',
                                    'enable_bias': True,
                                    'adj_type': 'corr',
                                    'enable_padding': True,
                                    'threshold': 0.3,
                                    'act_func': 'glu',
                                    'temporal_h_dim': 256,
                                    'spatial_h_dim': 64,
                                    'output_h_dim': 32,

                                    'weight_decay': 0.0780888601156736,
                                    'batch_size': 128,
                                    'lr': 0.00084,
                                    'dropout': 0.1702481119406736,
                                    'epochs': 500,

                                    'standardize': False,
                                    'minmaxnorm': True,
                                    },
                'subway_in_calendar_MSE_MinMax': {'target_data': 'subway_in',
                                                  'dataset_names': ['subway_in', 'calendar_embedding'],
                                                  'embedding_calendar_types': ['dayofweek', 'hour'],
                                                  'loss_function_type':'MSE',
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
                                                    },
                                                    
                    'subway_in_MSE_Huber_MinMax': {'target_data': 'subway_in',
                                          'dataset_names': ['subway_in'],
                                    'loss_function_type':'HuberLoss',
                                    'Kt': 2,
                                    'stblock_num': 2,
                                    'Ks': 2,
                                    'graph_conv_type': 'graph_conv',
                                    'gso_type': 'sym_renorm_adj',
                                    'enable_bias': True,
                                    'adj_type': 'corr',
                                    'enable_padding': True,
                                    'threshold': 0.3,
                                    'act_func': 'glu',
                                    'temporal_h_dim': 256,
                                    'spatial_h_dim': 64,
                                    'output_h_dim': 32,

                                    'weight_decay': 0.0780888601156736,
                                    'batch_size': 128,
                                    'lr': 0.00084,
                                    'dropout': 0.1702481119406736,
                                    'epochs': 500,

                                    'standardize': False,
                                    'minmaxnorm': True,
                                    },
                'subway_in_calendar_Huber_MinMax': {'target_data': 'subway_in',
                                                  'dataset_names': ['subway_in', 'calendar_embedding'],
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
                                                    },
                'subway_in_calendar_Huber_standardize': {'target_data': 'subway_in',
                                                  'dataset_names': ['subway_in', 'calendar_embedding'],
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,
                                                    },
                'subway_in_calendar_Google_Maps_Huber_standardize': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Google_Maps'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,
                                                    },
                'subway_in_calendar_Deezer_Huber_standardize': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Deezer'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,
                                                    },
                'subway_in_calendar_Web_Weather_Huber_standardize': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,
                                                    },
                'subway_in_calendar_Web_Weather_Google_Maps_Huber_standardize': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Google_Maps','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,
                                                    },
                'subway_in_calendar_Web_Weather_Deezer_Huber_standardize': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Deezer','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,
                                                    },
                'subway_in_calendar_Google_Maps_Deezer_Huber_standardize': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Deezer','Google_Maps'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,
                                                    },

                'subway_in_calendar_Google_Maps_Deezer_Huber_standardize_expsmooth07': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Deezer','Google_Maps'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,

                                                    'denoising_names':['netmob_POIs'],
                                                    'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                                    'denoising_modes':["train","valid","test"],             # par défaut
                                                    'denoiser_kwargs':{'exponential': {'alpha': 0.7}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
                                                    },

                'subway_in_calendar_Web_Weather_Deezer_Huber_standardize_expsmooth07': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Deezer','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,

                                                    'denoising_names':['netmob_POIs'],
                                                    'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                                    'denoising_modes':["train","valid","test"],             # par défaut
                                                    'denoiser_kwargs':{'exponential': {'alpha': 0.7}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
                                                    },
              'subway_in_calendar_Web_Weather_Google_Maps_Huber_standardize_expsmooth07': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Google_Maps','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,

                                                    'denoising_names':['netmob_POIs'],
                                                    'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                                    'denoising_modes':["train","valid","test"],             # par défaut
                                                    'denoiser_kwargs':{'exponential': {'alpha': 0.7}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
                                                    },

                'subway_in_calendar_Google_Maps_Deezer_Huber_standardize_expsmooth08': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Deezer','Google_Maps'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,

                                                    'denoising_names':['netmob_POIs'],
                                                    'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                                    'denoising_modes':["train","valid","test"],             # par défaut
                                                    'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
                                                    },

                'subway_in_calendar_Web_Weather_Deezer_Huber_standardize_expsmooth08': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Deezer','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,

                                                    'denoising_names':['netmob_POIs'],
                                                    'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                                    'denoising_modes':["train","valid","test"],             # par défaut
                                                    'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
                                                    },
              'subway_in_calendar_Web_Weather_Google_Maps_Huber_standardize_expsmooth08': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Google_Maps','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,

                                                    'denoising_names':['netmob_POIs'],
                                                    'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                                    'denoising_modes':["train","valid","test"],             # par défaut
                                                    'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
                                                    },

                'subway_in_calendar_Google_Maps_Deezer_Huber_standardize_expsmooth06': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Deezer','Google_Maps'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,

                                                    'denoising_names':['netmob_POIs'],
                                                    'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                                    'denoising_modes':["train","valid","test"],             # par défaut
                                                    'denoiser_kwargs':{'exponential': {'alpha': 0.6}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
                                                    },

                'subway_in_calendar_Web_Weather_Deezer_Huber_standardize_expsmooth06': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Deezer','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,

                                                    'denoising_names':['netmob_POIs'],
                                                    'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                                    'denoising_modes':["train","valid","test"],             # par défaut
                                                    'denoiser_kwargs':{'exponential': {'alpha': 0.6}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
                                                    },
              'subway_in_calendar_Web_Weather_Google_Maps_Huber_standardize_expsmooth06': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Google_Maps','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,

                                                    'denoising_names':['netmob_POIs'],
                                                    'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                                    'denoising_modes':["train","valid","test"],             # par défaut
                                                    'denoiser_kwargs':{'exponential': {'alpha': 0.6}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
                                                    },

                'subway_in_calendar_Google_Maps_Deezer_Huber_standardize_expsmooth09': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Deezer','Google_Maps'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,

                                                    'denoising_names':['netmob_POIs'],
                                                    'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                                    'denoising_modes':["train","valid","test"],             # par défaut
                                                    'denoiser_kwargs':{'exponential': {'alpha': 0.9}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
                                                    },

                'subway_in_calendar_Web_Weather_Deezer_Huber_standardize_expsmooth09': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Deezer','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,

                                                    'denoising_names':['netmob_POIs'],
                                                    'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                                    'denoising_modes':["train","valid","test"],             # par défaut
                                                    'denoiser_kwargs':{'exponential': {'alpha': 0.9}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
                                                    },
              'subway_in_calendar_Web_Weather_Google_Maps_Huber_standardize_expsmooth09': {'target_data': 'subway_in',
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
                                                    'standardize': True,
                                                    'minmaxnorm': False,

                                                    'NetMob_selected_apps'  :  ['Google_Maps','Web_Weather'], #,'Deezer','WhatsApp','Twitter', 'Google_Maps','Instagram'
                                                    'NetMob_transfer_mode' :  ['DL'], #,['UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['station_epsilon300'], 
                                                    'NetMob_expanded'  : '' ,# '' # '_expanded'
                                                    'NetMob_only_epsilon':  True ,

                                                    'denoising_names':['netmob_POIs'],
                                                    'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                                    'denoising_modes':["train","valid","test"],             # par défaut
                                                    'denoiser_kwargs':{'exponential': {'alpha': 0.9}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                                                                                                                  
                                                    },
 }

if len(constant_modif) > 0:
    modifications_bis = modifications.copy()
    modifications = {}
    for key, value in modifications_bis.items():
        modif_i = constant_modif.copy()
        modif_i.update(value)
        modifications[f"{constant_name}_{key}"] = modif_i