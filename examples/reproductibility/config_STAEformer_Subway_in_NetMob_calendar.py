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
constant_modif = {
    'loss_function_type':'HuberLoss',
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

                        'optimizer': 'adamw',
                        'batch_size': 128,
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h1_ldim1': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'denoising_names':[],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                                                                         'num_heads' : 1 ,
                        #                                                                                         'dim_model' : 48,}  
                        #                                                                         #'H' : ,
                        #                                                                         #'D': ,
                        #                                                                         #'W': , 
                        #                                                             },
                        #                                                     },  
                        #     },
                        # 'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h3_ldim1': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'denoising_names':[],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                         'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                         'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_HP0': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                                'denoising_names':[],
                                                                'lr': 0.00144143,
                                                                'weight_decay': 0.00685895,
                                                                'dropout': 0.1,
                                                                'torch_scheduler_milestone': 5,
                                                                'torch_scheduler_gamma': 0.988808,
                                                                'torch_scheduler_type': 'warmup',
                                                                'torch_scheduler_lr_start_factor': 0.499934,
                                                                'epochs':500,
                                                    'loss_function_type':'HuberLoss',
                                                    'optimizer': 'adamw',
                                                    'adaptive_embedding_dim': 32,
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

                                                    'input_embedding_dim': 24,
                                                    'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                                                                                                                'dim_model' : 64
                                                                                                                }  
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

                    'calendar_Google_Maps_Deezer_IRIS_input_dim12_attn_dim64_ff64_h2_ldim2_HP0': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                                'denoising_names':[],
                                                                'lr': 0.00144143,
                                                                'weight_decay': 0.00685895,
                                                                'dropout': 0.1,
                                                                'torch_scheduler_milestone': 5,
                                                                'torch_scheduler_gamma': 0.988808,
                                                                'torch_scheduler_type': 'warmup',
                                                                'torch_scheduler_lr_start_factor': 0.499934,
                                                                'epochs':500,
                                                    'loss_function_type':'HuberLoss',
                                                    'optimizer': 'adamw',
                                                    'adaptive_embedding_dim': 32,
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

                                                    'input_embedding_dim': 12,
                                                    'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                                                                                                                'dim_model' : 64
                                                                                                                }  
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

                 'calendar_Google_Maps_Deezer_IRIS_input_dim12_attn_dim64_ff64_h2_ldim2_HP2': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                                'denoising_names':[],
                                                                'lr': 0.00178889,
                                                                'weight_decay':0.00812294,
                                                                'dropout': 0.05   ,
                                                                'torch_scheduler_milestone': 5,
                                                                'torch_scheduler_gamma':  0.99079,
                                                                'torch_scheduler_type': 'warmup',
                                                                'torch_scheduler_lr_start_factor': 0.457083,
                                                                'epochs':500,
                                                    'loss_function_type':'HuberLoss',
                                                    'optimizer': 'adamw',
                                                    'adaptive_embedding_dim': 32,
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

                                                    'input_embedding_dim': 12,
                                                    'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                                                                                                                'dim_model' : 64
                                                                                                                }  
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

      'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_HP2': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                                'denoising_names':[],
                                                                'lr': 0.00178889,
                                                                'weight_decay':0.00812294,
                                                                'dropout': 0.05   ,
                                                                'torch_scheduler_milestone': 5,
                                                                'torch_scheduler_gamma':  0.99079,
                                                                'torch_scheduler_type': 'warmup',
                                                                'torch_scheduler_lr_start_factor': 0.457083,
                                                                'epochs':500,
                                                    'loss_function_type':'HuberLoss',
                                                    'optimizer': 'adamw',
                                                    'adaptive_embedding_dim': 32,
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

                                                    'input_embedding_dim': 24,
                                                    'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                                                                                                                'dim_model' : 64
                                                                                                                }  
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

      'calendar_Google_Maps_Deezer_IRIS_input_dim12_attn_dim64_ff64_h2_ldim2_HP3': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                                'denoising_names':[],
                                                                'lr':  0.00175035,
                                                                'weight_decay':0.00203832,
                                                                'dropout': 0.1   ,
                                                                'torch_scheduler_milestone': 5,
                                                                'torch_scheduler_gamma':   0.98754,
                                                                'torch_scheduler_type': 'warmup',
                                                                'torch_scheduler_lr_start_factor': 0.552248 ,
                                                                'epochs':500,

                                                    'loss_function_type':'HuberLoss',
                                                    'optimizer': 'adamw',
                                                    'adaptive_embedding_dim': 32,
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

                                                    'input_embedding_dim': 12,
                                                    'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                                                                                                                'dim_model' : 64
                                                                                                                }  
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


      'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_HP3': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                                'denoising_names':[],
                                                                'lr':  0.00175035,
                                                                'weight_decay':0.00203832,
                                                                'dropout': 0.1   ,
                                                                'torch_scheduler_milestone': 5,
                                                                'torch_scheduler_gamma':   0.98754,
                                                                'torch_scheduler_type': 'warmup',
                                                                'torch_scheduler_lr_start_factor': 0.552248 ,
                                                                'epochs':500,

                                                    'loss_function_type':'HuberLoss',
                                                    'optimizer': 'adamw',
                                                    'adaptive_embedding_dim': 32,
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

                                                    'input_embedding_dim': 24,
                                                    'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                                                                                                                'dim_model' : 64
                                                                                                                }  
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


      'calendar_Google_Maps_Deezer_IRIS_input_dim12_attn_dim64_ff64_h2_ldim2_HP4': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                                'denoising_names':[],
                                                                'lr':  0.00124175,
                                                                'weight_decay':0.00962615,
                                                                'dropout': 0.1   ,
                                                                'torch_scheduler_milestone': 5,
                                                                'torch_scheduler_gamma':   0.986409,
                                                                'torch_scheduler_type': 'warmup',
                                                                'torch_scheduler_lr_start_factor':0.346245 ,
                                                                'epochs':500,
                                                    'loss_function_type':'HuberLoss',
                                                    'optimizer': 'adamw',
                                                    'adaptive_embedding_dim': 32,
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

                                                    'input_embedding_dim': 12,
                                                    'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                                                                                                                'dim_model' : 64
                                                                                                                }  
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
                              'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_HP4': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                                'denoising_names':[],
                                                                'lr':  0.00124175,
                                                                'weight_decay':0.00962615,
                                                                'dropout': 0.1   ,
                                                                'torch_scheduler_milestone': 5,
                                                                'torch_scheduler_gamma':   0.986409,
                                                                'torch_scheduler_type': 'warmup',
                                                                'torch_scheduler_lr_start_factor':0.346245 ,
                                                                'epochs':500,
                                                    'loss_function_type':'HuberLoss',
                                                    'optimizer': 'adamw',
                                                    'adaptive_embedding_dim': 32,
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

                                                    'input_embedding_dim': 24,
                                                    'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                                                                                                                'dim_model' : 64
                                                                                                                }  
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


     'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim64_ff64_h2_ldim2_HP5': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                                'denoising_names':[],
                                                                'lr':  0.00250536 ,
                                                                'weight_decay':0.00192661,
                                                                'dropout': 0.0   ,
                                                                'torch_scheduler_milestone': 5,
                                                                'torch_scheduler_gamma':   0.985426,
                                                                'torch_scheduler_type': 'warmup',
                                                                'torch_scheduler_lr_start_factor':0.628773 ,
                                                                'epochs':500,

                                                    'loss_function_type':'HuberLoss',
                                                    'optimizer': 'adamw',
                                                    'adaptive_embedding_dim': 32,
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

                                                    'input_embedding_dim': 24,
                                                    'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                                                                                                                'dim_model' : 64
                                                                                                                }  
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

 'calendar_Google_Maps_Deezer_IRIS_input_dim12_attn_dim64_ff64_h2_ldim2_HP5': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                                'denoising_names':[],
                                                                'lr':  0.00250536 ,
                                                                'weight_decay':0.00192661,
                                                                'dropout': 0.0   ,
                                                                'torch_scheduler_milestone': 5,
                                                                'torch_scheduler_gamma':   0.985426,
                                                                'torch_scheduler_type': 'warmup',
                                                                'torch_scheduler_lr_start_factor':0.628773 ,
                                                                'epochs':500,

                                                    'loss_function_type':'HuberLoss',
                                                    'optimizer': 'adamw',
                                                    'adaptive_embedding_dim': 32,
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

                                                    'input_embedding_dim': 12,
                                                    'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                                                                                                                'dim_model' : 64
                                                                                                                }  
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        # 'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim96_ff128_h3_ldim4_exp_smooth_08': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                        #                                 'input_embedding_dim': 12*2,
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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
                        #                                 'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
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