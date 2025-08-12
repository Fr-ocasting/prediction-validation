import pandas as pd  # if not, I get this error while running a .py from terminal: 
# ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found (required by /root/anaconda3/envs/pytorch-2.0.1_py-3.10.5/lib/python3.10/site-packages/pandas/_libs/window/aggregations.cpython-310-x86_64-linux-gnu.so)
import torch 
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True

# Relative path:
import sys 
import os 
current_path = os.path.abspath(os.path.dirname(__file__))
working_dir = os.path.abspath(os.path.join(current_path,'..'))
#parent_dir = f"{parent_dir}/prediction_validation/"  # A utiliser sur .ipynb notebook
if working_dir not in sys.path:
    sys.path.insert(0,working_dir)
# ...

# Personnal import 
from examples.HP_parameter_choice import hyperparameter_tuning
from examples.train_model_on_k_fold_validation import train_model_on_k_fold_validation,load_configuration
import numpy as np 

def HP_and_valid_one_config(args,epochs_validation,num_samples):
    # HP Tuning on the first fold
    analysis,trial_id = hyperparameter_tuning(args,num_samples)

    # K-fold validation with best config: 
    modification = {'epochs':epochs_validation}
    train_model_on_k_fold_validation(trial_id,load_config=True,save_folder='K_fold_validation/training_with_HP_tuning',modification=modification)
    return trial_id


def set_one_hp_tuning_and_evaluate_DA(args=None,epochs_validation=None,num_samples=None):

    # HP tuning and return the trial-id : 
    trial_id = HP_and_valid_one_config(args,epochs_validation,num_samples)

    #trial_id = 'subway_in_subway_out_STGCN_VariableSelectionNetwork_MSELoss_2025_01_20_05_38_87836'
    #trial_id = 'subway_in_STGCN_MSELoss_2025_01_17_18_25_95152'  -> 


    save_folder = 'K_fold_validation/training_with_HP_tuning/re_validation'

    if True:
        modification ={'keep_best_weights':True,
                        'epochs':epochs_validation,
                        }

        config_diffs = {'maps_deezer_insta_DL_iris_rich_interpolation':{'DA_method' : ['rich_interpolation'],
                        'data_augmentation' : True
                          },                                         
                        'maps_deezer_insta_DL_iris_magnitude_warping0075':{'DA_method' : ['magnitude_warping'],
                        'data_augmentation' : True,
                        'DA_magnitude_max_scale':0.075
                                                },
                        'maps_deezer_insta_DL_iris_rich_interpolation_and_magnitude_warping0075':{'DA_method' : ['rich_interpolation','magnitude_warping'],
                        'data_augmentation' : True,
                        'DA_magnitude_max_scale':0.075
                                                }
                                    }


        if False:
            config_diffs = {'DA_Homogenous_1':{'data_augmentation': True, #True,  #False
                            'DA_method':'noise', # 'noise' # 'interpolation
                            'DA_min_count': 5,
                            'DA_alpha' : 1,
                            'DA_prop' : 1, # 1 #0.005
                            'DA_noise_from': 'Homogenous' # 'MSTL' # 'Homogenous'
                            },
                        'DA_Homogenous_0.2':{'data_augmentation': True, #True,  #False
                            'DA_method':'noise', # 'noise' # 'interpolation
                            'DA_min_count': 5,
                            'DA_alpha' : 0.2,
                            'DA_prop' : 1, # 1 #0.005
                            'DA_noise_from': 'Homogenous' # 'MSTL' # 'Homogenous'
                            },
                        'DA_MSTL_02':{'data_augmentation': True, #True,  #False
                            'DA_method':'noise', # 'noise' # 'interpolation
                            'DA_min_count': 5,
                            'DA_alpha' : 0.2,
                            'DA_prop' : 1, # 1 #0.005
                            'DA_noise_from': 'MSTL' # 'MSTL' # 'Homogenous'
                            },
                        'DA_MSTL_1':{'data_augmentation': True, #True,  #False
                            'DA_method':'noise', # 'noise' # 'interpolation
                            'DA_min_count': 5,
                            'DA_alpha' : 1,
                            'DA_prop' : 1, # 1 #0.005
                            'DA_noise_from': 'MSTL' # 'MSTL' # 'Homogenous'
                            },
                        'No_DA':{'data_augmentation': False, #True,  #False
                            },
                        'DA_interpolation':{'data_augmentation': True, #True,  #False
                            'DA_method':'interpolation', # 'noise' # 'interpolation
                            }
                        }
 
    if False:
        modification ={'keep_best_weights':True,
                        'epochs':1,
                        'DA_moment_to_focus' : None
                        }


        config_diffs = {'lr1':{'data_augmentation': False, #True,  #False
                        'DA_method':'noise', # 'noise' # 'interpolation
                        'lr':0.1},
                    'lr2':{'data_augmentation': False, #True,  #False
                        'DA_method':'noise', # 'noise' # 'interpolation
                        'lr':0.01},
                    'lr3':{'data_augmentation': False, #True,  #False
                                            'DA_method':'noise', # 'noise' # 'interpolation
                                            'lr':0.001}
                    }

                  
    for add_name_id,config_diff in config_diffs.items():
        config_diff.update(modification)
        train_model_on_k_fold_validation(trial_id,load_config =True,
                                         save_folder=save_folder,
                                         modification=config_diff,
                                         add_name_id=add_name_id)


if __name__ == '__main__':

    #from file00 import *
    #vision_model_name = 'FeatureExtractorEncoderDecoder'  # 'ImageAvgPooling'  # 'FeatureExtractor_ResNetInspired_bis'  #'FeatureExtractor_ResNetInspired' #'MinimalFeatureExtractor',
    # 'AttentionFeatureExtractor' # 'FeatureExtractorEncoderDecoder' # 'VideoFeatureExtractorWithSpatialTemporalAttention'
    from examples.benchmark import local_get_args

    if True:
        model_name = 'STGCN' #'CNN'
        target_data = 'subway_in'
        dataset_for_coverage = [target_data,'netmob_POIs'] 
        dataset_names = [target_data,'calendar_embedding','netmob_POIs']
        args = local_get_args(model_name,
                            args_init = None,
                            dataset_names=dataset_names,
                            dataset_for_coverage=dataset_for_coverage,
                            modification = {'target_data' :target_data,
                                            'ray':True,
                                            'grace_period':20,
                                            'HP_max_epochs':1000, #1000, #300,
                                            'K_fold': 2,
                                            'evaluate_complete_ds' : True,
                                            'embedding_calendar_types': ['dayofweek', 'hour'],


                                            # Architecture 
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

                                            # Optimization
                                            'loss_function_type':'HuberLoss',
                                            'optimizer': 'adamw',
                                            'batch_size': 128,
                                            'freq': '15min',
                                            'weight_decay': 0.0014517707449388,
                                            'batch_size': 128,
                                            'lr': 0.00071,
                                            'dropout': 0.145169206052754,
                                            'H':6,
                                            'D':1,
                                            'W':0,
                                            'step_ahead': 1,
                                            'horizon_step' : 1,
                                            'unormalize_loss' : True,
                                            'torch_scheduler': None,


                                            'temporal_graph_transformer_encoder': False, # False # True
                                            'need_global_attn' : False, # False # True
                                            'data_augmentation': False, #True,  #False
                                            'use_target_as_context': False,


                                            
                                            'standardize': False,
                                            'minmaxnorm': True,

                                            'TE_embedding_dim': 64,
                                            'TE_out_h_dim': 64,
                                            'TE_concatenation_late': True,
                                            'TE_concatenation_early':False,

                                            'contextual_kwargs' : {'netmob_POIs': {'need_global_attn':True, 
                                                                                    'stacked_contextual': False,
                                                                                    'NetMob_selected_apps' : ['Google_Maps','Web_Weather'], # Google_Maps # 
                                                                                    'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                                    'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                                    'NetMob_expanded' : '', # '' # '_expanded'
                                                                                    'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                                                                    'vision_model_name' : None,
                                                                                    'epsilon_clustering': 0.1,
                                                                                    'agg_iris_target_n': None,
                                                                                    'use_only_for_common_dates': False, # If True then only use the dataset to restrain Feature vector to the common dates between the datasets
                                                                                    'attn_kwargs': {
                                                                                                    'dim_feedforward' : 128,
                                                                                                    'num_heads' : 4 ,
                                                                                                    'dim_model' : 64,
                                                                                                    'keep_topk':30,
                                                                                                    'nb_layers': 3,
                                                                                                    'latent_dim': 64,
                                                                        
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

                                            'num_workers' : 0, #4, # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
                                            'persistent_workers' : False ,# True 
                                            'pin_memory' : False ,# True 
                                            'prefetch_factor' : None, # 4, # None, 2,3,4,5 ... 
                                            'drop_last' : False,  # True
                                            'mixed_precision' : False, # True # False
                                            'torch_compile' : False,# 'compile', # 'compile' # 'jit_script' #'trace'

                                             })


        epochs_validation = 1000 #1000
        num_samples = 200#200
        HP_and_valid_one_config(args,epochs_validation,num_samples)

        

    # if True:
    #     model_name = 'DCRNN' #'CNN'
    #     target_data = 'subway_in'
    #     dataset_for_coverage = [target_data,'netmob_POIs'] 
    #     dataset_names = [target_data,'calendar_embedding']
    #     args = local_get_args(model_name,
    #                         args_init = None,
    #                         dataset_names=dataset_names,
    #                         dataset_for_coverage=dataset_for_coverage,
    #                         modification = {'target_data' :target_data,
    #                                         'ray':True,
    #                                         'grace_period':10,
    #                                         'HP_max_epochs':1,#500,
    #                                         'K_fold': 2,
    #                                         'evaluate_complete_ds' : True,
    #                                         'vision_model_name': None,
    #                                         'stacked_contextual': False, # True # False

    #                                         # Preprocess
    #                                         'standardize': False,
    #                                         'minmaxnorm': True,
    #                                         'data_augmentation': False, #True,  #False

    #                                         # Other Module: 
    #                                         'need_global_attn' : False, # False # True
    #                                         'use_target_as_context': False,
    #                                         'temporal_graph_transformer_encoder': False, # False # True

    #                                         # Optim
    #                                         'optimizer': 'adamw',
    #                                         'loss_function_type':'HuberLoss',
    #                                         'torch_scheduler': None,
    #                                         'batch_size': 128,
    #                                         'step_ahead': 4,
    #                                         'freq': '15min',
    #                                         'H':6,
    #                                         'D':1,
    #                                         'W':0,

    #                                         # Time Embedding: 
    #                                         'TE_embedding_dim': 64,
    #                                         'TE_out_h_dim': 64,
    #                                         'TE_multi_embedding': True,
    #                                         'TE_concatenation_late' : True,
    #                                         'TE_concatenation_early' : False,
    #                                         'TE_variable_selection_model_name': 'MLP',
    #                                         'embedding_calendar_types': ['dayofweek', 'hour'],

    #                                         })

    #     # Init 
    #     epochs_validation =1 # 500
    #     num_samples = 2 #300
    #     HP_and_valid_one_config(args,epochs_validation,num_samples)
    #     #set_one_hp_tuning_and_evaluate_DA(args,epochs_validation,num_samples)
        


    if False:
        model_name = 'RNN' #'CNN'
        target_data = 'subway_in'
        dataset_for_coverage = [target_data,'netmob_POIs'] 
        dataset_names = [target_data,'calendar_embedding']
        args = local_get_args(model_name,
                            args_init = None,
                            dataset_names=dataset_names,
                            dataset_for_coverage=dataset_for_coverage,
                            modification = {'target_data' :target_data,
                                            'ray':True,
                                            'grace_period':40,
                                            'HP_max_epochs':1000, #300,
                                            'K_fold': 2,
                                            'evaluate_complete_ds' : True,
                                            'vision_model_name': None,
                                            'stacked_contextual': False, # True # False

                                            # Preprocess
                                            'standardize': False,
                                            'minmaxnorm': True,
                                            'data_augmentation': False, #True,  #False

                                            # Other Module: 
                                            'need_global_attn' : False, # False # True
                                            'use_target_as_context': False,
                                            'temporal_graph_transformer_encoder': False, # False # True

                                            # Optim
                                            'optimizer': 'adamw',
                                            'loss_function_type':'HuberLoss',
                                            'torch_scheduler': None,
                                            'batch_size': 128,
                                            'step_ahead': 4,
                                            'freq': '15min',
                                            'H':6,
                                            'D':1,
                                            'W':0,

                                            # Time Embedding: 
                                            'TE_embedding_dim': 64,
                                            'TE_out_h_dim': 64,
                                            'TE_multi_embedding': True,
                                            'TE_concatenation_late' : True,
                                            'TE_concatenation_early' : False,
                                            'TE_variable_selection_model_name': 'MLP',
                                            'embedding_calendar_types': ['dayofweek', 'hour'],

                                            # # Computation Ressources: 
                                            # 'num_workers' : 4, #4, # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
                                            # 'persistent_workers' : True ,# False 
                                            # 'pin_memory' : True ,# False 
                                            # 'prefetch_factor' : 4, # None, 2,3,4,5 ... 
                                            # 'drop_last' : False,  # True
                                            # 'mixed_precision' : False, # True # False
                                            # #'torch_compile' : 'compile', # 'compile' # 'jit_script' #'trace'

                                            #  
                                            })

        # Init 
        epochs_validation = 1000#300
        num_samples = 300#200
        HP_and_valid_one_config(args,epochs_validation,num_samples)
        #set_one_hp_tuning_and_evaluate_DA(args,epochs_validation,num_samples)
        
    if False:
        model_name = 'STGCN' #'CNN'
        target_data = 'PeMS08_flow'
        dataset_for_coverage = [target_data] 
        dataset_names = [target_data,'calendar_embedding']
        args = local_get_args(model_name,
                            args_init = None,
                            dataset_names=dataset_names,
                            dataset_for_coverage=dataset_for_coverage,
                            modification = {'target_data' :target_data,
                                            'ray':True,
                                            'grace_period':10,
                                            'HP_max_epochs':300, #300,
                                            'K_fold': 2,
                                            'evaluate_complete_ds' : True,
                                            'vision_model_name': None,
                                            'stacked_contextual': False, # True # False
                                            'temporal_graph_transformer_encoder': False, # False # True
                                            'need_global_attn' : False, # False # True
                                            'data_augmentation': False, #True,  #False

                                            'use_target_as_context': False,
                                            'data_augmentation': False,

                                            'loss_function_type':'HuberLoss',
                                            'torch_scheduler': None,

                                            'step_ahead': 12,
                                            'freq': '5min',
                                            'H':12,
                                            'D':0,
                                            'W':0,
                                            
                                            'standardize': True,
                                            'minmaxnorm': False,

                                            'Kt': 3, # 2,3,4 # Kernel Size on the Temporal Dimension
                                            'stblock_num': 3, # 2,3,4 # Number of STConv-blocks
                                            'Ks': 3,  # 1,2,3 # Number of iteration within the ChebGraphConv ONLY
                                            'graph_conv_type': 'cheb_graph_conv', # 'graph_conv','cheb_graph_conv' # Type of graph convolution
                                            'gso_type': 'sym_renorm_adj', # 'sym_norm_lap','rw_norm_lap','sym_renorm_adj','rw_renorm_adj'  # Type of calcul to compute the gso (Weighted Adjacency Matrix)
                                            'temporal_h_dim': 128,
                                            'spatial_h_dim': 64,
                                            'output_h_dim': 256,
                                            'optimizer': 'adamw',
                                            'adj_type': 'corr',
                                            'threshold': 0.8,
                                            'enable_bias': True, # Enable bias on the output module (FC layers at the output of STGCN)
                                            'enable_padding': True,  # Enable padding on the Temporal convolution. Suitable for short sequence cause (L' = L-2*(Kt-1)*stblock_num)
                                            'act_func': 'glu', #'glu', 'gtu', 'silu'  # Type of activation function on the output module (FC layers at the output of STGCN)  
                                            'batch_size': 128,
                                            'epochs':300,

                                            'TE_embedding_dim': 64,
                                            'TE_multi_embedding': True,
                                            'TE_concatenation_late' : True,
                                            'TE_concatenation_early' : False,
                                            'TE_out_h_dim': 32,
                                            'TE_variable_selection_model_name': 'MLP',
                                            'TE_embedding_calendar_types' : ['dayofweek', 'hour','minute'],  # ['dayofweek', 'hour', 'minute']

                                            'num_workers' : 4, #4, # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
                                            'persistent_workers' : True ,# False 
                                            'pin_memory' : True ,# False 
                                            'prefetch_factor' : 4, # None, 2,3,4,5 ... 
                                            'drop_last' : False,  # True
                                            'mixed_precision' : False, # True # False
                                            #'torch_compile' : 'compile', # 'compile' # 'jit_script' #'trace'

                                             })

        # Init 
        epochs_validation = 300#300
        num_samples = 200#200
        HP_and_valid_one_config(args,epochs_validation,num_samples)
        #set_one_hp_tuning_and_evaluate_DA(args,epochs_validation,num_samples)
        
    if False:

        #model_name = 'ASTGCN' #'CNN' # 'STGCN' # ASTGCN # STGformer
        dataset_for_coverage = ['subway_in','netmob_POIs'] 
        model_name = 'STAEformer'

        epochs_validation =300# 500
        num_samples = 300 # 500
        HP_max_epochs = 300 # 500
        modification  = {'ray':True,
                        'target_data' :'subway_in',
                        'use_target_as_context': False,

                        'batch_size':128,
                        'grace_period':20,#20,
                        'HP_max_epochs':HP_max_epochs,
                        'step_ahead':4,

                        'evaluate_complete_ds' : True,
                        'torch_compile':False,

                        'temporal_graph_transformer_encoder': False, # False # True
                        'need_global_attn' : False, # False # True
                        'stacked_contextual': True, # True # False

                        'data_augmentation': True, #True,  #False
                        'DA_method':'rich_interpolation', # 'noise' # 'interpolation

                        'denoising_names':['netmob_POIs'],
                        'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        'denoising_modes':["train","valid","test"],             # par défaut
                        'denoiser_kwargs':{'exponential': {'alpha': 0.7}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        }

        modif_choices = {'no_netmob':{'dataset_names' : ['subway_in','calendar']}}
        for name_i,modif_bis in modif_choices.items(): 
            modif_bis.update(modification)
            args = local_get_args(model_name,
                                args_init = None,
                                dataset_names=modif_bis['dataset_names'],
                                dataset_for_coverage=dataset_for_coverage,
                                modification =modif_bis
                                )
            args.calendar_types = ['dayofweek', 'timeofday']
            HP_and_valid_one_config(args,epochs_validation,num_samples)

    if False:

        #model_name = 'ASTGCN' #'CNN' # 'STGCN' # ASTGCN # STGformer
        dataset_for_coverage = ['subway_in','netmob_POIs'] 
        model_name = 'STAEformer'

        epochs_validation =500# 500
        num_samples = 300 # 500
        HP_max_epochs = 500 # 500
        modification  = {'ray':True,
                        'target_data' :'subway_in',
                        'use_target_as_context': False,

                        'batch_size':128,
                        'grace_period':20,#20,
                        'HP_max_epochs':HP_max_epochs,
                        'step_ahead':4,

                        'evaluate_complete_ds' : True,
                        'torch_compile':False,

                        'temporal_graph_transformer_encoder': False, # False # True
                        'need_global_attn' : False, # False # True

                        'data_augmentation': False, #True,  #False
                        'DA_method':'rich_interpolation', # 'noise' # 'interpolation

                        'K_fold': 2,
                        }

        modif_choices = {'Google_Maps_Deezer_IRIS': {'dataset_names': ['subway_in','calendar','netmob_POIs'],
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


            # 'no_netmob':{'dataset_names' : ['subway_in','calendar']},
            
        # 'weather':{'dataset_names' : ['subway_in','netmob_POIs','calendar'],
        #         'NetMob_only_epsilon': True,
        #         'NetMob_selected_apps': ['Web_Weather'],
        #         'NetMob_transfer_mode' :  ['DL'],
        #         'NetMob_selected_tags' : ['station_epsilon300'],
        #         'NetMob_expanded' : ''},



        # 'Deezer':{'dataset_names' : ['subway_in','netmob_POIs','calendar'],
        #             'NetMob_only_epsilon': True,
        #             'NetMob_selected_apps': ['Deezer'],
        #             'NetMob_transfer_mode' :  ['DL'],
        #             'NetMob_selected_tags' : ['station_epsilon300'],
        #             'NetMob_expanded' : ''},

        # 'Google_Maps':{'dataset_names' : ['subway_in','netmob_POIs','calendar'],
        #                 'NetMob_only_epsilon': True,
        #                 'NetMob_selected_apps': ['Google_Maps'],
        #                 'NetMob_transfer_mode' :  ['DL'],
        #                 'NetMob_selected_tags' : ['station_epsilon300'],
        #                 'NetMob_expanded' : ''},

        # 'weather_deezer':{'dataset_names' : ['subway_in','netmob_POIs','calendar'],
        #                 'NetMob_only_epsilon': True,
        #                 'NetMob_selected_apps': ['Web_Weather','Deezer'],
        #                 'NetMob_transfer_mode' :  ['DL'],
        #                 'NetMob_selected_tags' : ['station_epsilon300'],
        #                 'NetMob_expanded' : ''},

        # 'Google_Maps_deezer':{'dataset_names' : ['subway_in','netmob_POIs','calendar'],
        #                     'NetMob_only_epsilon': True,
        #                     'NetMob_selected_apps': ['Google_Maps','Deezer'],
        #                     'NetMob_transfer_mode' :  ['DL'],
        #                     'NetMob_selected_tags' : ['station_epsilon300'],
        #                     'NetMob_expanded' : ''},

        # 'Google_Maps_weather':{'dataset_names' : ['subway_in','netmob_POIs','calendar'],
        #                         'NetMob_only_epsilon': True,
        #                         'NetMob_selected_apps': ['Google_Maps','Web_Weather'],
        #                         'NetMob_transfer_mode' :  ['DL'],
        #                         'NetMob_selected_tags' : ['station_epsilon300'],
        #                         'NetMob_expanded' : ''},

        # 'Deezer_Google_Maps_weather':{'dataset_names' : ['subway_in','netmob_POIs','calendar'],
        #                             'NetMob_only_epsilon': True,
        #                             'NetMob_selected_apps': ['Deezer','Google_Maps','Web_Weather'],
        #                             'NetMob_transfer_mode' :  ['DL'],
        #                             'NetMob_selected_tags' : ['station_epsilon300'],
        #                             'NetMob_expanded' : ''},

        }
        for name_i,modif_bis in modif_choices.items(): 
            modif_bis.update(modification)
            args = local_get_args(model_name,
                                args_init = None,
                                dataset_names=modif_bis['dataset_names'],
                                dataset_for_coverage=dataset_for_coverage,
                                modification =modif_bis
                                )
            args.calendar_types = ['dayofweek', 'timeofday']
            HP_and_valid_one_config(args,epochs_validation,num_samples)


    if False:

        #model_name = 'ASTGCN' #'CNN' # 'STGCN' # ASTGCN # STGformer
        dataset_for_coverage = ['subway_in','netmob_POIs'] 
        model_name = 'STGCN'

        epochs_validation = 500#100
        num_samples = 200 # 200
        HP_max_epochs = 500 #300,#100,
        modification  = {'ray':True,
                        'target_data' :'subway_in',
                        'use_target_as_context': False,

                        'batch_size':128,
                        'grace_period':20,#20,
                        'HP_max_epochs':HP_max_epochs,
                        'step_ahead':4,

                        'evaluate_complete_ds' : True,
                        'torch_compile':False,

                        'temporal_graph_transformer_encoder': False, # False # True
                        'need_global_attn' : False, # False # True
                        'stacked_contextual': True, # True # False

                        'data_augmentation': True, #True,  #False
                        'DA_method':'rich_interpolation', # 'noise' # 'interpolation
                        }

        modif_choices = {'no_netmob_no_calendar':{'dataset_names' : ['subway_in']},

                        'weather':{'dataset_names' : ['subway_in','netmob_POIs'],
                                'NetMob_only_epsilon': True,
                                'NetMob_selected_apps': ['Web_Weather'],
                                'NetMob_transfer_mode' :  ['DL'],
                                'NetMob_selected_tags' : ['station_epsilon300'],
                                'NetMob_expanded' : ''},



                        'Deezer':{'dataset_names' : ['subway_in','netmob_POIs'],
                                    'NetMob_only_epsilon': True,
                                    'NetMob_selected_apps': ['Deezer'],
                                    'NetMob_transfer_mode' :  ['DL'],
                                    'NetMob_selected_tags' : ['station_epsilon300'],
                                    'NetMob_expanded' : ''},

                        'Google_Maps':{'dataset_names' : ['subway_in','netmob_POIs'],
                                        'NetMob_only_epsilon': True,
                                        'NetMob_selected_apps': ['Google_Maps'],
                                        'NetMob_transfer_mode' :  ['DL'],
                                        'NetMob_selected_tags' : ['station_epsilon300'],
                                        'NetMob_expanded' : ''},

                        'weather_deezer':{'dataset_names' : ['subway_in','netmob_POIs'],
                                        'NetMob_only_epsilon': True,
                                        'NetMob_selected_apps': ['Web_Weather','Deezer'],
                                        'NetMob_transfer_mode' :  ['DL'],
                                        'NetMob_selected_tags' : ['station_epsilon300'],
                                        'NetMob_expanded' : ''},

                        'Google_Maps_deezer':{'dataset_names' : ['subway_in','netmob_POIs'],
                                            'NetMob_only_epsilon': True,
                                            'NetMob_selected_apps': ['Google_Maps','Deezer'],
                                            'NetMob_transfer_mode' :  ['DL'],
                                            'NetMob_selected_tags' : ['station_epsilon300'],
                                            'NetMob_expanded' : ''},

                        'Google_Maps_weather':{'dataset_names' : ['subway_in','netmob_POIs'],
                                                'NetMob_only_epsilon': True,
                                                'NetMob_selected_apps': ['Google_Maps','Web_Weather'],
                                                'NetMob_transfer_mode' :  ['DL'],
                                                'NetMob_selected_tags' : ['station_epsilon300'],
                                                'NetMob_expanded' : ''},

                        'Deezer_Google_Maps_weather':{'dataset_names' : ['subway_in','netmob_POIs'],
                                                    'NetMob_only_epsilon': True,
                                                    'NetMob_selected_apps': ['Deezer','Google_Maps','Web_Weather'],
                                                    'NetMob_transfer_mode' :  ['DL'],
                                                    'NetMob_selected_tags' : ['station_epsilon300'],
                                                    'NetMob_expanded' : ''},
                                                    
                        'no_netmob_with_calendar':{'dataset_names' : ['subway_in','calendar_embedding'],
                                                                    'embedding_calendar_types': ['dayofweek', 'hour'],}
                                        }
        for name_i,modif_bis in modif_choices.items(): 
            modif_bis.update(modification)
            args = local_get_args(model_name,
                                args_init = None,
                                dataset_names=modif_bis['dataset_names'],
                                dataset_for_coverage=dataset_for_coverage,
                                modification =modif_bis
                                )
            HP_and_valid_one_config(args,epochs_validation,num_samples)

