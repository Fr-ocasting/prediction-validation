import pandas as pd  # if not, I get this error while running a .py from terminal: 
# ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found (required by /root/anaconda3/envs/pytorch-2.0.1_py-3.10.5/lib/python3.10/site-packages/pandas/_libs/window/aggregations.cpython-310-x86_64-linux-gnu.so)

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
        vision_model_name = 'VariableSelectionNetwork'  # None #'VariableSelectionNetwork'
        args = local_get_args(model_name,
                                args_init = None,
                                dataset_names=dataset_names,
                                dataset_for_coverage=dataset_for_coverage,
                                modification = {'ray':True,
                                                'grace_period':20,
                                                'HP_max_epochs':100,
                                                'evaluate_complete_ds' : True,
                                                'vision_model_name': vision_model_name,
                                                'DA_method' : ['rich_interpolation'],
                                                'data_augmentation' : True,
                                                })

        # Init 
        epochs_validation = 100
        num_samples = 500
        #HP_and_valid_one_config(args,epochs_validation,num_samples)
        set_one_hp_tuning_and_evaluate_DA(args,epochs_validation,num_samples)
    if True:
        model_name = 'STGCN' #'CNN'
        dataset_for_coverage = ['subway_in','netmob_POIs'] 
        dataset_names = ['subway_in','subway_out'] # ['subway_in','netmob_POIs_per_station']
        vision_model_name = None #'VariableSelectionNetwork'

        args = local_get_args(model_name,
                            args_init = None,
                            dataset_names=dataset_names,
                            dataset_for_coverage=dataset_for_coverage,
                            modification = {'ray':True,
                                            'grace_period':20,#20,
                                            'HP_max_epochs':100,#100,
                                            'evaluate_complete_ds' : True,
                                            #'set_spatial_units' : ['BON','SOI','GER','CHA'],
                                            'stacked_contextual': True, # True # False
                                            'temporal_graph_transformer_encoder': True, # False # True
                                            'compute_node_attr_with_attn' : False, # False # True

                                            'vision_model_name': None,

                                            'data_augmentation': True, #True,  #False
                                            'DA_method':'rich_interpolation', # 'noise' # 'interpolation
                                            })
        # Init 
        epochs_validation = 100#100
        num_samples = 500
        
        HP_and_valid_one_config(args,epochs_validation,num_samples)
    if False:
        model_name = 'STGCN' #'CNN'
        dataset_for_coverage = ['subway_in','netmob_POIs'] 
        for dataset_names,vision_model_name in zip([['subway_in','netmob_POIs']], #['subway_in','subway_out'] # ['subway_in']
                                                   ['VariableSelectionNetwork']): #'VariableSelectionNetwork' # None
            args = local_get_args(model_name,
                                    args_init = None,
                                    dataset_names=dataset_names,
                                    dataset_for_coverage=dataset_for_coverage,
                                    modification = {'ray':True,
                                                    'grace_period':20,
                                                    'HP_max_epochs':100,
                                                    'evaluate_complete_ds' : True,
                                                    'vision_model_name': vision_model_name,
                                                   }
                                    
                                     )

            # Init 
            epochs_validation = 100
            num_samples = 500

            # HP and evaluate K-fold best config
            HP_and_valid_one_config(args,epochs_validation,num_samples)