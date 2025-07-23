constant_name = ''
constant_modif = {}
SEED = 1
config =  { 'target_data': 'subway_in',
            'dataset_names': ['subway_in', 'calendar_embedding','netmob_POIs'],
            'dataset_for_coverage': ['subway_in', 'netmob_POIs'],
            'embedding_calendar_types': ['dayofweek', 'hour'],
             'use_target_as_context': False,
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
            'freq': '15min',
            'H':6,
            'D':1,
            'W':0,
            'unormalize_loss' : True,
             'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                                                    'stacked_contextual': True,
                                                    'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
                                                    'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                    'NetMob_expanded' : '', # '' # '_expanded'
                                                    'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                    'epsilon_clustering': None, 
                                                    'vision_model_name' : None,
                                                    'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                    'attn_kwargs': {'latent_dim': 2 ,
                                                                    'dim_feedforward' : 64,
                                                                    'num_heads':  2 ,
                                                                    'dim_model':  64,
                                                                    'keep_topk': True}  
                                                    #'H' : ,
                                                    #'D': ,
                                                    #'W': , 
                                        },
                                    },  
            'denoising_names':['netmob_POIs'],
            'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
            'denoising_modes':["train","valid","test"],             # par défaut
            'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
            }
