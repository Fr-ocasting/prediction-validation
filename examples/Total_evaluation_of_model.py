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

    if False:
        model_name = 'STGCN' #'CNN'
        dataset_for_coverage = ['subway_in','netmob_POIs'] 
        dataset_names = ['subway_in','netmob_POIs']
        args = local_get_args(model_name,
                                args_init = None,
                                dataset_names=dataset_names,
                                dataset_for_coverage=dataset_for_coverage,
                                modification = {'ray':True,
                                                'grace_period':20,
                                                'HP_max_epochs':100,
                                                'evaluate_complete_ds' : True,
                                                'vision_model_name': None,
                                                'stacked_contextual': True, # True # False
                                                'temporal_graph_transformer_encoder': False, # False # True
                                                'compute_node_attr_with_attn' : False, # False # True
                                                'data_augmentation': True, #True,  #False
                                                'DA_method':'rich_interpolation', # 'noise' # 'interpolation
                                                'NetMob_selected_apps':  ['Deezer'], # 'Google_Maps'
                                                'NetMob_transfer_mode' :  ['DL'],
                                                'NetMob_selected_tags': ['station_epsilon100'],  #'station_epsilon300'
                                                'NetMob_expanded' : '',
                                                })

        # Init 
        epochs_validation = 100
        num_samples = 100
        HP_and_valid_one_config(args,epochs_validation,num_samples)
        #set_one_hp_tuning_and_evaluate_DA(args,epochs_validation,num_samples)
  if True:

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
                        'compute_node_attr_with_attn' : False, # False # True
                        'stacked_contextual': True, # True # False

                        'data_augmentation': True, #True,  #False
                        'DA_method':'rich_interpolation', # 'noise' # 'interpolation

                        'denoising_names':['netmob_POIs'],
                        'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        'denoising_modes':["train","valid","test"],             # par défaut
                        'denoiser_kwargs':{'exponential': {'alpha': 0.7}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        }

        modif_choices = {'no_netmob':{'dataset_names' : ['subway_in','calendar']},

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

    if True:

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
                        'compute_node_attr_with_attn' : False, # False # True
                        'stacked_contextual': True, # True # False

                        'data_augmentation': True, #True,  #False
                        'DA_method':'rich_interpolation', # 'noise' # 'interpolation

                        'denoising_names':['netmob_POIs'],
                        'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                        'denoising_modes':["train","valid","test"],             # par défaut
                        'denoiser_kwargs':{'exponential': {'alpha': 0.7}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}
                        }

        modif_choices = {'no_netmob':{'dataset_names' : ['subway_in','calendar']},
            
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

        'weather_deezer':{'dataset_names' : ['subway_in','netmob_POIs','calendar'],
                        'NetMob_only_epsilon': True,
                        'NetMob_selected_apps': ['Web_Weather','Deezer'],
                        'NetMob_transfer_mode' :  ['DL'],
                        'NetMob_selected_tags' : ['station_epsilon300'],
                        'NetMob_expanded' : ''},

        'Google_Maps_deezer':{'dataset_names' : ['subway_in','netmob_POIs','calendar'],
                            'NetMob_only_epsilon': True,
                            'NetMob_selected_apps': ['Google_Maps','Deezer'],
                            'NetMob_transfer_mode' :  ['DL'],
                            'NetMob_selected_tags' : ['station_epsilon300'],
                            'NetMob_expanded' : ''},

        'Google_Maps_weather':{'dataset_names' : ['subway_in','netmob_POIs','calendar'],
                                'NetMob_only_epsilon': True,
                                'NetMob_selected_apps': ['Google_Maps','Web_Weather'],
                                'NetMob_transfer_mode' :  ['DL'],
                                'NetMob_selected_tags' : ['station_epsilon300'],
                                'NetMob_expanded' : ''},

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
                        'compute_node_attr_with_attn' : False, # False # True
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

