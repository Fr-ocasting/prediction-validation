# Modif CRITER - STGCN

constant_name = ''
constant_modif = {'dataset_for_coverage': ['CRITER_3_4_5_lanes_flow','netmob_POIs'],
                  'target_data': 'CRITER_3_4_5_lanes_flow',
                  'model_name': 'STGCN',
                        'optimizer': 'adamw',
                        'batch_size': 128,
                        'epochs': 1000,
                        'step_ahead' : 10,
                        'horizon_step' : 5,
                        'freq': '6min',
                        'H':10,
                        'D':0,
                        'W':0,
                        }


modifications = { 'Init': {'dataset_names': ['CRITER_3_4_5_lanes_flow','netmob_POIs'],
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
                              'standardize': False,
                              'minmaxnorm': True,

                              'contextual_kwargs' :  {'netmob_POIs': {'need_global_attn':True, 
                                                                                                'stacked_contextual': True,
                                                                                                'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
                                                                                                'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                                'NetMob_expanded' : '', # '' # '_expanded'
                                                                                                'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                                'epsilon_clustering': 0.15, 
                                                                                                'vision_model_name' : None,
                                                                                                'use_only_for_common_dates': True, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                                'attn_kwargs': {'latent_dim': 2 ,
                                                                                                                'dim_feedforward' : 128,
                                                                                                                'num_heads':  3 ,
                                                                                                                'dim_model':  48,
                                                                                                                'keep_topk': False}  
                                                                                                #'H' : ,
                                                                                                #'D': ,
                                                                                                #'W': , 
                                                                                    },
                                                                                }, 
                              'denoising_names':[],        
                                                },
                    'calendar': {'dataset_names': ['CRITER_3_4_5_lanes_flow', 'calendar_embedding','netmob_POIs'],
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
                                  'standardize': False,
                                  'minmaxnorm': True,

                                  'TE_embedding_dim': 64,
                                  'TE_out_h_dim': 64,
                                  'TE_concatenation_late': True,
                                  'TE_concatenation_early':False,
                                  'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                                                                                                'stacked_contextual': True,
                                                                                                'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
                                                                                                'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                                'NetMob_expanded' : '', # '' # '_expanded'
                                                                                                'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                                'epsilon_clustering': 0.15, 
                                                                                                'vision_model_name' : None,
                                                                                                'use_only_for_common_dates': True, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                                'attn_kwargs': {'latent_dim': 2 ,
                                                                                                                'dim_feedforward' : 128,
                                                                                                                'num_heads':  3 ,
                                                                                                                'dim_model':  48,
                                                                                                                'keep_topk': False}  
                                                                                                #'H' : ,
                                                                                                #'D': ,
                                                                                                #'W': , 
                                                                                    },
                                                                                }, 
                                  'denoising_names':[],        
                                                },
'calendar_Google_Maps_Deezer_IRIS_clustering015': {'dataset_names': ['CRITER_3_4_5_lanes_flow', 'calendar_embedding','netmob_POIs'],
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
                                                        'standardize': False,
                                                        'minmaxnorm': True,

                                                        'TE_embedding_dim': 64,
                                                        'TE_out_h_dim': 64,
                                                        'TE_concatenation_late': True,
                                                        'TE_concatenation_early':False,
                                                        'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                                                                                                'stacked_contextual': True,
                                                                                                'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
                                                                                                'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                                'NetMob_expanded' : '', # '' # '_expanded'
                                                                                                'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                                'epsilon_clustering': 0.15, 
                                                                                                'vision_model_name' : None,
                                                                                                'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                                'attn_kwargs': {'latent_dim': 2 ,
                                                                                                                'dim_feedforward' : 128,
                                                                                                                'num_heads':  3 ,
                                                                                                                'dim_model':  48,
                                                                                                                'keep_topk': False}  
                                                                                                #'H' : ,
                                                                                                #'D': ,
                                                                                                #'W': , 
                                                                                    },
                                                                                },  
                                                        'denoising_names':[],        
                                                },
                                        
}


if len(constant_modif) > 0:
    modifications_bis = modifications.copy()
    modifications = {}
    for key, value in modifications_bis.items():
        modif_i = constant_modif.copy()
        modif_i.update(value)
        modifications[f"{constant_name}_{key}"] = modif_i