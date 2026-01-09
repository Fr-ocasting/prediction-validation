import copy
# Description of Contextual Data Integration strategies: 
# Early fusion 
# --- Shared Embedding
#       concatenate contextual data on channel dim BEFORE embedding of input block 
# --- Independant Embedding
#       concatenate contextual data on channel dim AFTER separate embedding target and of contextual 
# --- Feature Extractor
# Late fusion
# --- Simple Embedding
# --- Feature Extraxtor
# --- Traffic Model backbone



AGG_IRIS_DEFAULT_N = 100
THRESHOLD_VOLUME_MIN_DEFAULT = 1
QUANTILE_FILTER_OUTLIERS_DEFAULT = 0.99

INPUT_EMBEDDING_DIM = 48
REPEAT_TRIAL = 5

possible_target_kwargs = {
    'subway_out': {'quantile_filter_outliers': QUANTILE_FILTER_OUTLIERS_DEFAULT  },  
     'subway_in': { 'quantile_filter_outliers': QUANTILE_FILTER_OUTLIERS_DEFAULT }, 
     'bike_out': {'agg_iris_target_n':AGG_IRIS_DEFAULT_N,
                  'threshold_volume_min': THRESHOLD_VOLUME_MIN_DEFAULT},
     'bike_in': {'agg_iris_target_n':AGG_IRIS_DEFAULT_N,
                  'threshold_volume_min': THRESHOLD_VOLUME_MIN_DEFAULT},
     'PeMS08_flow': {},
      }


feature_extractor_model_configurations = {'need_global_attn':True, 
                                        'stacked_contextual': False,
                                        'vision_model_name' : None,
                                        'use_only_for_common_dates': False,
                                        'quantile_filter_outliers':QUANTILE_FILTER_OUTLIERS_DEFAULT ,
                                        'attn_kwargs': {
                                            'input_embedding_dim': INPUT_EMBEDDING_DIM, 
                                            # 'init_adaptive_query_dim' : 24,    # Do not use Target data as query 
                                            'adaptive_embedding_dim' : 32,     # Add two separate adaptive Embedding within Query and Key/Values.
                                            'feed_forward_dim':256, 
                                            'num_heads':4,
                                            'num_layers':3,
                                            'mask':False,
                                            'keep_temporal_dim': True,
                                            'tod_embedding_dim' : 6,
                                            'dow_embedding_dim': 6,
                                                         },
                                         }  

model_configurations = {
    'STAEformer': {   'calendar_types':['dayofweek', 'timeofday'],
                        'input_embedding_dim': INPUT_EMBEDDING_DIM, 
                        'adaptive_embedding_dim': 32,
                        'tod_embedding_dim': 6,
                        'dow_embedding_dim': 6,
                        'feed_forward_dim': 256,
                        'num_heads': 4,
                        'num_layers': 3,
                        'use_mixed_proj': True,

                        'lr': 0.001,
                        'weight_decay':  0.0015,
                        'dropout': 0.2,
                        'torch_scheduler_milestone': 20,
                        'torch_scheduler_gamma':0.9925,
                        'torch_scheduler_type': 'warmup',
                        'torch_scheduler_lr_start_factor': 0.3,

                        'standardize': False,
                        'minmaxnorm': True,
                        'H':6,
                        'D':1,
                        'W':0,
                        'batch_size': 128,
                        'epochs':200,
    },
    'STGCN': {
        'use_target_as_context':False, 

        'embedding_calendar_types': ['dayofweek', 'hour'],
        'TE_embedding_dim': 64,
        'TE_out_h_dim': 64,
        'TE_concatenation_late': True,
        'TE_concatenation_early':False,

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
        'H':6,
        'D':1,
        'W':0,

        'batch_size': 128,
        'epochs': 200,
    },
} 



netmob_preprocessing_kwargs = {
    'denoiser_kwargs': {'exponential': {'alpha': 0.8}},
    'contextual_kwargs': {'netmob_POIs': {
                                        'NetMob_selected_apps': ['Google_Maps', 'Web_Weather'], # 'Deezer' # Instagram'
                                        'NetMob_transfer_mode': ['DL'],
                                        'NetMob_selected_tags': ['iris'],
                                        'NetMob_expanded': '',
                                        'NetMob_only_epsilon': False,
                                        'epsilon_clustering': 0.1,
                                        'agg_iris_target_n': None,
                                        }
                        }
    }




subway_possible_contextual_kwargs = {

           'late_fusion': {'adp_query_cross_attn_traffic_model_backbone' : copy.deepcopy(feature_extractor_model_configurations),
                           
                           'cross_attn_traffic_model_backbone' : copy.deepcopy(feature_extractor_model_configurations),
                            
                            'traffic_model_backbone':copy.deepcopy(feature_extractor_model_configurations),

                            'feature_extractor':copy.deepcopy(feature_extractor_model_configurations),
                                        
                            'simple_embedding':{ 'need_global_attn':True, 
                                                    'stacked_contextual': False,
                                                    'vision_model_name' : None,
                                                    'use_only_for_common_dates': False,
                                                    'quantile_filter_outliers': QUANTILE_FILTER_OUTLIERS_DEFAULT ,

                                                    'attn_kwargs': {
                                                        'simple_embedding_dim': INPUT_EMBEDDING_DIM,
                                                        'concatenation_late': True,
                                                        },
                                                },
                    

                                },


             'early_fusion': { 
                                'adp_query_cross_attn_traffic_model_backbone' : copy.deepcopy(feature_extractor_model_configurations),

                                'cross_attn_traffic_model_backbone' : copy.deepcopy(feature_extractor_model_configurations),
                                
                                'traffic_model_backbone':copy.deepcopy(feature_extractor_model_configurations),

                                'feature_extractor': copy.deepcopy(feature_extractor_model_configurations),      


                                's_proj_t_proj': {'emb_dim' : INPUT_EMBEDDING_DIM,
                                                                'need_global_attn':False, 
                                                                'stacked_contextual': False,
                                                                'vision_model_name' : None,
                                                                'use_only_for_common_dates': False,
                                                                'quantile_filter_outliers': QUANTILE_FILTER_OUTLIERS_DEFAULT ,
                                                                'unique_serie': False,   # If true then agregate all (2) weather stations into one unique serie
                                                                'repeat_spatial': False,  # If true then repeat the weather serie for each node of the target data
                                                                'attn_kwargs': {'concatenation_late':False},
                                                            }, 


                                'independant_embedding':{'emb_dim' : INPUT_EMBEDDING_DIM,
                                        'need_global_attn':False, 
                                        'stacked_contextual': False,
                                        'vision_model_name' : None,
                                        'use_only_for_common_dates': False,
                                        'quantile_filter_outliers': QUANTILE_FILTER_OUTLIERS_DEFAULT ,
                                        'attn_kwargs': {},
                                        },


                                'shared_embedding':{  
                                                    'need_global_attn':False, 
                                                    'stacked_contextual': True,
                                                    'vision_model_name' : None,
                                                    'use_only_for_common_dates': False,
                                                    'quantile_filter_outliers': QUANTILE_FILTER_OUTLIERS_DEFAULT ,
                                                    'attn_kwargs': {},
                                                    },

                                        },

         

                    }



# --- Late Fusions : 
subway_possible_contextual_kwargs['late_fusion']['feature_extractor']['attn_kwargs']['concatenation_late'] = True

subway_possible_contextual_kwargs['late_fusion']['traffic_model_backbone']['backbone_model'] = True
subway_possible_contextual_kwargs['late_fusion']['traffic_model_backbone']['attn_kwargs']['concatenation_late'] = True

subway_possible_contextual_kwargs['late_fusion']['cross_attn_traffic_model_backbone']['attn_kwargs']['cross_attention'] = True
subway_possible_contextual_kwargs['late_fusion']['cross_attn_traffic_model_backbone']['backbone_model'] = True
subway_possible_contextual_kwargs['late_fusion']['cross_attn_traffic_model_backbone']['attn_kwargs']['concatenation_late'] = True

subway_possible_contextual_kwargs['late_fusion']['adp_query_cross_attn_traffic_model_backbone']['attn_kwargs']['cross_attention'] = True
subway_possible_contextual_kwargs['late_fusion']['adp_query_cross_attn_traffic_model_backbone']['backbone_model'] = True
subway_possible_contextual_kwargs['late_fusion']['adp_query_cross_attn_traffic_model_backbone']['attn_kwargs']['init_adaptive_query_dim'] = INPUT_EMBEDDING_DIM
subway_possible_contextual_kwargs['late_fusion']['adp_query_cross_attn_traffic_model_backbone']['attn_kwargs']['concatenation_late'] = True
# ---

# --- Early Fusions : 
subway_possible_contextual_kwargs['late_fusion']['feature_extractor']['attn_kwargs']['concatenation_late'] = False

subway_possible_contextual_kwargs['early_fusion']['traffic_model_backbone']['backbone_model'] = True
subway_possible_contextual_kwargs['early_fusion']['traffic_model_backbone']['attn_kwargs']['concatenation_late'] = False

subway_possible_contextual_kwargs['early_fusion']['cross_attn_traffic_model_backbone']['attn_kwargs']['cross_attention'] = True
subway_possible_contextual_kwargs['early_fusion']['cross_attn_traffic_model_backbone']['backbone_model'] = True
subway_possible_contextual_kwargs['early_fusion']['cross_attn_traffic_model_backbone']['attn_kwargs']['concatenation_late'] = False

subway_possible_contextual_kwargs['early_fusion']['adp_query_cross_attn_traffic_model_backbone']['attn_kwargs']['cross_attention'] = True
subway_possible_contextual_kwargs['early_fusion']['adp_query_cross_attn_traffic_model_backbone']['backbone_model'] = True
subway_possible_contextual_kwargs['early_fusion']['adp_query_cross_attn_traffic_model_backbone']['attn_kwargs']['init_adaptive_query_dim'] = INPUT_EMBEDDING_DIM
subway_possible_contextual_kwargs['early_fusion']['adp_query_cross_attn_traffic_model_backbone']['attn_kwargs']['concatenation_late'] = False
# ---


bike_possible_contextual_kwargs = {
                    'early_fusion': {'independant_embedding':{'emb_dim' : INPUT_EMBEDDING_DIM,
                                                                'need_global_attn':False, 
                                                                'stacked_contextual': False,
                                                                'vision_model_name' : None,
                                                                'use_only_for_common_dates': False,
                                                                'agg_iris_target_n':AGG_IRIS_DEFAULT_N,
                                                                'threshold_volume_min': THRESHOLD_VOLUME_MIN_DEFAULT,
                                                                'quantile_filter_outliers': QUANTILE_FILTER_OUTLIERS_DEFAULT,
                                                                'attn_kwargs': {},
                                                                },

                                    'shared_embedding':{
                                                        'need_global_attn':False, 
                                                        'stacked_contextual': True,
                                                        'vision_model_name' : None,
                                                        'use_only_for_common_dates': False,
                                                        'quantile_filter_outliers': QUANTILE_FILTER_OUTLIERS_DEFAULT ,
                                                        'attn_kwargs': {},
                                                        },  

                                    'feature_extractor': copy.deepcopy(feature_extractor_model_configurations),         
                                    },

                    'late_fusion': {'simple_embedding':{'emb_dim' : 12,
                                                        },  

                                    'feature_extractor':copy.deepcopy(feature_extractor_model_configurations),   

                                    'traffic_model_backbone':{
                                                            },
                    },
}

weather_possible_contextual_kwargs = {

                    'late_fusion': {  'feature_extractor':copy.deepcopy(feature_extractor_model_configurations),

                                       's_proj_t_proj': {'emb_dim' : 8,
                                                            'need_global_attn':False, 
                                                            'stacked_contextual': False,
                                                            'vision_model_name' : None,
                                                            'use_only_for_common_dates': False,
                                                            'quantile_filter_outliers': QUANTILE_FILTER_OUTLIERS_DEFAULT ,
                                                            'unique_serie': True,   # If true then agregate all (2) weather stations into one unique serie
                                                            'repeat_spatial': False,  # If true then repeat the weather serie for each node of the target data
                                                            'attn_kwargs': {'concatenation_late':True},
                                                           }, 

                                                           
                                        'repeat_t_proj': {'emb_dim' : 8,
                                                            'need_global_attn':False, 
                                                            'stacked_contextual': False,
                                                            'vision_model_name' : None,
                                                            'use_only_for_common_dates': False,
                                                            'quantile_filter_outliers': QUANTILE_FILTER_OUTLIERS_DEFAULT ,
                                                            'unique_serie': True,   # If true then agregate all (2) weather stations into one unique serie
                                                            'repeat_spatial': True,  # If true then repeat the weather serie for each node of the target data
                                                            'attn_kwargs': {'concatenation_late':True},
                                                      },  
                                    }, 

                    'early_fusion': {         

                                     'repeat_t_proj': {'emb_dim' : 8,
                                                            'need_global_attn':False, 
                                                            'stacked_contextual': False,
                                                            'vision_model_name' : None,
                                                            'use_only_for_common_dates': False,
                                                            'quantile_filter_outliers': QUANTILE_FILTER_OUTLIERS_DEFAULT ,
                                                            'unique_serie': True,   # If true then agregate all (2) weather stations into one unique serie
                                                            'repeat_spatial': True,  # If true then repeat the weather serie for each node of the target data
                                                            'attn_kwargs': {},
                                                         }, 
                                        's_proj_t_proj': {'emb_dim' : 8,
                                                            'need_global_attn':False, 
                                                            'stacked_contextual': False,
                                                            'vision_model_name' : None,
                                                            'use_only_for_common_dates': False,
                                                            'quantile_filter_outliers': QUANTILE_FILTER_OUTLIERS_DEFAULT ,
                                                            'unique_serie': True,   # If true then agregate all (2) weather stations into one unique serie
                                                            'repeat_spatial': False,  # If true then repeat the weather serie for each node of the target data
                                                            'attn_kwargs': {},
                                                         }, 

                                     'feature_extractor':copy.deepcopy(feature_extractor_model_configurations),  
                                    }, 

                        }

# ---- Modify Feature Extractor parameter for weather data: (make it simple) ----
weather_possible_contextual_kwargs['early_fusion']['feature_extractor']['unique_serie'] = True
weather_possible_contextual_kwargs['early_fusion']['feature_extractor']['attn_kwargs']['concatenation_late'] = False

weather_possible_contextual_kwargs['early_fusion']['feature_extractor']['attn_kwargs']['input_embedding_dim'] = 8
weather_possible_contextual_kwargs['early_fusion']['feature_extractor']['attn_kwargs']['feed_forward_dim'] = 64
weather_possible_contextual_kwargs['early_fusion']['feature_extractor']['attn_kwargs']['num_heads'] = 4
weather_possible_contextual_kwargs['early_fusion']['feature_extractor']['attn_kwargs']['num_layers'] = 1
weather_possible_contextual_kwargs['early_fusion']['feature_extractor']['attn_kwargs']['tod_embedding_dim'] = 4 # 0
weather_possible_contextual_kwargs['early_fusion']['feature_extractor']['attn_kwargs']['dow_embedding_dim'] = 4 # 0
weather_possible_contextual_kwargs['early_fusion']['feature_extractor']['attn_kwargs']['adaptive_embedding_dim'] = 4 # 0


weather_possible_contextual_kwargs['late_fusion']['feature_extractor']['attn_kwargs']['concatenation_late'] = True
weather_possible_contextual_kwargs['late_fusion']['feature_extractor']['unique_serie'] = True

weather_possible_contextual_kwargs['late_fusion']['feature_extractor']['attn_kwargs']['input_embedding_dim'] = 8
weather_possible_contextual_kwargs['late_fusion']['feature_extractor']['attn_kwargs']['feed_forward_dim'] = 64
weather_possible_contextual_kwargs['late_fusion']['feature_extractor']['attn_kwargs']['num_heads'] = 4
weather_possible_contextual_kwargs['late_fusion']['feature_extractor']['attn_kwargs']['num_layers'] = 1
weather_possible_contextual_kwargs['late_fusion']['feature_extractor']['attn_kwargs']['tod_embedding_dim'] = 4 # 0
weather_possible_contextual_kwargs['late_fusion']['feature_extractor']['attn_kwargs']['dow_embedding_dim'] = 4 # 0
weather_possible_contextual_kwargs['late_fusion']['feature_extractor']['attn_kwargs']['adaptive_embedding_dim'] = 4 # 0


# weather_possible_contextual_kwargs['early_fusion']['feature_extractor']['attn_kwargs']['adaptive_embedding_dim'] = 0 # 32 
# weather_possible_contextual_kwargs['early_fusion']['feature_extractor']['attn_kwargs']['fusion_type'] = 'sum' # 'concat'



percent_train_size = {
    '5p': {'calib_prop' : 0.95},
    '10p': {'calib_prop' : 0.90},
    '15p': {'calib_prop' : 0.85},
    '25p':  {'calib_prop' : 0.75},
    # '35p':  {'calib_prop' : 0.65},
    '50p':  {'calib_prop' : 0.5},
    '75p':  {'calib_prop' : 0.25},
    # '80p':  {'calib_prop' : 0.2},
    # '85p':  {'calib_prop' : 0.15},
    # '90p':  {'calib_prop' : 0.1},
    # '95p':  {'calib_prop' : 0.05},
    '100p': {'calib_prop': None},
     }


expanding_train_size = {
    '5p': {'expanding_train' : 0.05},
    '10p': {'expanding_train' : 0.10},
    '15p': {'expanding_train' : 0.15},
    '25p':  {'expanding_train' : 0.25},
    # '35p':  {'calib_prop' : 0.65},
    '50p':  {'expanding_train' : 0.5},
    '75p':  {'expanding_train' : 0.75},
    # '80p':  {'calib_prop' : 0.2},
    # '85p':  {'calib_prop' : 0.15},
    # '90p':  {'calib_prop' : 0.1},
    # '95p':  {'calib_prop' : 0.05},
    '100p': {'calib_prop': None},
     }

def modif_percent_train_size(target_data,freq,percent,modif_percent,dataset_for_coverage):
    if (
        ((target_data in ['bike_out','bike_in','subway_in','subway_out']) and (freq == '1H'))
        or 
        ('netmob_POIs' in dataset_for_coverage)
        ):
        if int(percent[:-1]) <= 50:
            if int(percent[:-1]) > 0: 
                modif_percent['batch_size'] = 16
            elif int(percent[:-1]) > 25:
                modif_percent['batch_size'] = 32
            elif int(percent[:-1]) >= 50:
                modif_percent['batch_size'] = 64
            else:
                raise ValueError(f"Percent {percent} not expected")
    else:
        if int(percent[:-1]) == 5:
            modif_percent['batch_size'] = 32
    return modif_percent