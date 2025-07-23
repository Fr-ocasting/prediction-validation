constant_name = ''
constant_modif = {}
SEED = 1
config = {'target_data': 'subway_in',
        'dataset_names': ['subway_in','calendar','netmob_POIs'],
        'dataset_for_coverage': ['subway_in', 'netmob_POIs'],
        'use_target_as_context': False,
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
                                                'NetMob_selected_apps' : ['Deezer','Google_Maps'], # Google_Maps # 
                                                'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                'NetMob_expanded' : '', # '' # '_expanded'
                                                'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                'vision_model_name' : None,
                                                'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                'attn_kwargs': {'latent_dim' : 2 ,
                                                                'dim_feedforward' : 64,
                                                                'num_heads' : 2 ,
                                                                'dim_model' : 12,}  
                                                #'H' : ,
                                                #'D': ,
                                                #'W': , 
                                        },
                                },  
        'denoising_names':['netmob_POIs'],
        'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
        'denoising_modes':["train","valid","test"],             # par défaut
        'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}``
}