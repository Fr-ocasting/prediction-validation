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


                        'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h3_ldim2': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                        'denoising_names':[],
                                                        'input_embedding_dim': 12*2,
                                                        'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
                                                                                                'stacked_contextual': True,
                                                                                                'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                                                                                                'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                                'NetMob_expanded' : '', # '' # '_expanded'
                                                                                                'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                                'vision_model_name' : None,
                                                                                                'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                                'attn_kwargs': {'latent_dim' : 2 ,
                                                                                                                'dim_feedforward' : 64,
                                                                                                                'num_heads' : 3 ,
                                                                                                                'dim_model' : 48,}  
                                                                                                #'H' : ,
                                                                                                #'D': ,
                                                                                                #'W': , 
                                                                                    },
                                                                            },  
                            },
                        'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h1_ldim4': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                        'denoising_names':[],
                                                        'input_embedding_dim': 12*2,
                                                        'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
                                                                                                'stacked_contextual': True,
                                                                                                'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                                                                                                'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                                'NetMob_expanded' : '', # '' # '_expanded'
                                                                                                'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                                'vision_model_name' : None,
                                                                                                'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                                'attn_kwargs': {'latent_dim' : 4 ,
                                                                                                                'dim_feedforward' : 64,
                                                                                                                'num_heads' : 1 ,
                                                                                                                'dim_model' : 48,}  
                                                                                                #'H' : ,
                                                                                                #'D': ,
                                                                                                #'W': , 
                                                                                    },
                                                                            },  
                            },

                        'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h1_ldim2': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                        'denoising_names':[],
                                                        'input_embedding_dim': 12*2,
                                                        'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
                                                                                                'stacked_contextual': True,
                                                                                                'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                                                                                                'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                                'NetMob_expanded' : '', # '' # '_expanded'
                                                                                                'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                                'vision_model_name' : None,
                                                                                                'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                                'attn_kwargs': {'latent_dim' : 2 ,
                                                                                                                'dim_feedforward' : 64,
                                                                                                                'num_heads' : 1 ,
                                                                                                                'dim_model' : 48,}  
                                                                                                #'H' : ,
                                                                                                #'D': ,
                                                                                                #'W': , 
                                                                                    },
                                                                            },  
                            },
                        'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h1_ldim1': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                        'denoising_names':[],
                                                        'input_embedding_dim': 12*2,
                                                        'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
                                                                                                'stacked_contextual': True,
                                                                                                'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                                                                                                'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                                'NetMob_expanded' : '', # '' # '_expanded'
                                                                                                'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                                'vision_model_name' : None,
                                                                                                'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                                'attn_kwargs': {'latent_dim' : 1 ,
                                                                                                                'dim_feedforward' : 64,
                                                                                                                'num_heads' : 1 ,
                                                                                                                'dim_model' : 48,}  
                                                                                                #'H' : ,
                                                                                                #'D': ,
                                                                                                #'W': , 
                                                                                    },
                                                                            },  
                            },
                        'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h3_ldim1': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                        'denoising_names':[],
                                                        'input_embedding_dim': 12*2,
                                                        'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
                                                                                                'stacked_contextual': True,
                                                                                                'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                                                                                                'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                                'NetMob_expanded' : '', # '' # '_expanded'
                                                                                                'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                                'vision_model_name' : None,
                                                                                                'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                                'attn_kwargs': {'latent_dim' : 1 ,
                                                                                                                'dim_feedforward' : 64,
                                                                                                                'num_heads' : 3 ,
                                                                                                                'dim_model' : 48,}  
                                                                                                #'H' : ,
                                                                                                #'D': ,
                                                                                                #'W': , 
                                                                                    },
                                                                            },  
                            },

                        'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h2_ldim4': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                        'denoising_names':[],
                                                        'input_embedding_dim': 12*2,
                                                        'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
                                                                                                'stacked_contextual': True,
                                                                                                'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                                                                                                'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                                'NetMob_expanded' : '', # '' # '_expanded'
                                                                                                'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                                'vision_model_name' : None,
                                                                                                'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                                'attn_kwargs': {'latent_dim' : 4 ,
                                                                                                                'dim_feedforward' : 64,
                                                                                                                'num_heads' : 2 ,
                                                                                                                'dim_model' : 48,}  
                                                                                                #'H' : ,
                                                                                                #'D': ,
                                                                                                #'W': , 
                                                                                    },
                                                                            },  
                            },

                        'calendar_Deezer_IRIS_input_dim24_attn_dim48_ff64_h3_ldim4': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                        'denoising_names':[],
                                                        'input_embedding_dim': 12*2,
                                                        'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
                                                                                                'stacked_contextual': True,
                                                                                                'NetMob_selected_apps' : ['Deezer'], # Google_Maps # 
                                                                                                'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                                'NetMob_expanded' : '', # '' # '_expanded'
                                                                                                'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                                'vision_model_name' : None,
                                                                                                'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                                'attn_kwargs': {'latent_dim' : 4 ,
                                                                                                                'dim_feedforward' : 64,
                                                                                                                'num_heads' : 3 ,
                                                                                                                'dim_model' : 48,}  
                                                                                                #'H' : ,
                                                                                                #'D': ,
                                                                                                #'W': , 
                                                                                    },
                                                                            },  
                            },

                           'calendar_Google_Maps_IRIS_input_dim24_attn_dim48_ff64_h3_ldim4': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                                'denoising_names':[],
                                                               'input_embedding_dim': 12*2,
                                                        'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
                                                                                                'stacked_contextual': True,
                                                                                                'NetMob_selected_apps' : ['Google_Maps'], # Google_Maps # 
                                                                                                'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                                'NetMob_expanded' : '', # '' # '_expanded'
                                                                                                'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                                'vision_model_name' : None,
                                                                                                'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                                'attn_kwargs': {'latent_dim' : 4 ,
                                                                                                                'dim_feedforward' : 64,
                                                                                                                'num_heads' : 3 ,
                                                                                                                'dim_model' : 48,}  
                                                                                                #'H' : ,
                                                                                                #'D': ,
                                                                                                #'W': , 
                                                                                    },
                                                                            },   
                                                      },

                         'calendar_Google_Maps_Deezer_IRIS_input_dim24_attn_dim48_ff64_h3_ldim4': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
                                                                'denoising_names':[],
                                                                 'input_embedding_dim': 12*2,
                                                                'contextual_kwargs' : {'netmob_POIs': {'compute_node_attr_with_attn':True, 
                                                                                                        'stacked_contextual': True,
                                                                                                        'NetMob_selected_apps' : ['Google_Maps','Deezer'], # Google_Maps # 
                                                                                                        'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                                        'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                                        'NetMob_expanded' : '', # '' # '_expanded'
                                                                                                        'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                                        'vision_model_name' : None,
                                                                                                        'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                                        'attn_kwargs': {'latent_dim' : 4 ,
                                                                                                                        'dim_feedforward' : 64,
                                                                                                                        'num_heads' : 3 ,
                                                                                                                        'dim_model' : 48,}  
                                                                                                        #'H' : ,
                                                                                                        #'D': ,
                                                                                                        #'W': , 
                                                                                            },
                                                                                    },   
                            },               


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