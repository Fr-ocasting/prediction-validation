
# GET PARAMETERS
import os 
import sys
import torch 
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from examples.train_model_on_k_fold_validation import train_model_on_k_fold_validation,load_configuration


if False: 
    save_folder = 'K_fold_validation/training_with_HP_tuning/re_validation'

    trial_id = 'subway_in_STGCN_MSELoss_2025_01_20_14_27_20569'
    epochs_validation = 100
    args,folds = load_configuration(trial_id,True)

    modification ={'keep_best_weights':True,
                    'epochs':epochs_validation,
                    'device':torch.device("cuda:0"),
                    }


    config_diffs = {'magnitude_warping_010':{'dataset_names':['subway_in'],
                                            'data_augmentation': True,
                                            'DA_method':'magnitude_warping',
                                            'DA_magnitude_max_scale':0.10,
                                            },
                    'magnitude_warping_0125':{'dataset_names':['subway_in'],
                                            'data_augmentation': True,
                                            'DA_method':'magnitude_warping',
                                            'DA_magnitude_max_scale':0.125,
                                            },
                    'magnitude_warping_015':{'dataset_names':['subway_in'],
                                            'data_augmentation': True,
                                            'DA_method':'magnitude_warping',
                                            'DA_magnitude_max_scale':0.15,
                                            },
                }

                        
    for add_name_id,config_diff in config_diffs.items():
        config_diff.update(modification)
        train_model_on_k_fold_validation(trial_id,load_config =True,
                                            save_folder=save_folder,
                                            modification=config_diff,
                                            add_name_id=add_name_id)



if True: 

    save_folder = 'K_fold_validation/training_with_HP_tuning/re_validation'
    trial_id = 'subway_in_subway_out_STGCN_VariableSelectionNetwork_MSELoss_2025_01_20_05_38_87836'
    epochs_validation = 100
    args,folds = load_configuration(trial_id,True)

    modification ={'keep_best_weights':True,
                    'epochs':epochs_validation,
                    'device':torch.device("cuda:0"),
                    'loss_function_type':'quantile',
                    'alpha':0.05,
                    'track_pi':True,
                    'type_calib':'classic',
                    }



    config_diffs = {'UQ_rich_interpolation':{'dataset_names':['subway_in','subway_out'],
                                            'vision_concatenation_early' : True,
                                            'vision_concatenation_late' : False,
                                            'data_augmentation': True,
                                            'DA_method':'rich_interpolation',
                                            },
                }

                        
    for add_name_id,config_diff in config_diffs.items():
        config_diff.update(modification)
        train_model_on_k_fold_validation(trial_id,load_config =True,
                                            save_folder=save_folder,
                                            modification=config_diff,
                                            add_name_id=add_name_id)