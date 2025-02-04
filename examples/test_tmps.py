
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


if True: 
    save_folder = 'K_fold_validation/training_with_HP_tuning/re_validation'
    trial_id = 'subway_in_subway_out_STGCN_VariableSelectionNetwork_MSELoss_2025_01_20_05_38_87836'
    epochs_validation = 100
    args,folds = load_configuration(trial_id,True)

    modification ={'keep_best_weights':True,
                    'epochs':epochs_validation,
                    'device':torch.device("cuda:0"),
                    }

    config_diffs = {'large_attention_DA_rich_interpolation_nh8_emb64':{'dataset_names':['subway_in','subway_out'],
                                            'data_augmentation': True,
                                            'DA_method':'rich_interpolation',
                                            'freq':'15min',
                                            'vision_concatenation_early' : True,
                                            'vision_concatenation_late' : False,
                                            'vision_grn_out_dim':64,
                                            'vision_num_heads':8,
                                            #'NetMob_selected_apps':  ['Google_Maps','Instagram','Deezer'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                                            #'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                            #'NetMob_selected_tags' : ['iris','station'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                            #'NetMob_expanded' : '_expanded', # '' # '_expanded'
                                            #'data_augmentation': True,
                                            #'DA_method':'magnitude_warping',
                                            },
                    'large_attention_DA_rich_interpolation_nh8_emb128':{'dataset_names':['subway_in','subway_out'],
                                            'data_augmentation': True,
                                            'DA_method':'rich_interpolation',
                                            'freq':'15min',
                                            'vision_concatenation_early' : True,
                                            'vision_concatenation_late' : False,
                                            'vision_grn_out_dim':128,
                                            'vision_num_heads':8,
                                            },
                    'large_attention_DA_rich_interpolation_nh3_emb64':{'dataset_names':['subway_in','subway_out'],
                                            'data_augmentation': True,
                                            'DA_method':'rich_interpolation',
                                            'freq':'15min',
                                            'vision_concatenation_early' : True,
                                            'vision_concatenation_late' : False,
                                            'vision_grn_out_dim':64,
                                            'vision_num_heads':3,
                                            },
                    'large_attention_DA_rich_interpolation_nh3_emb32':{'dataset_names':['subway_in','subway_out'],
                                            'data_augmentation': True,
                                            'DA_method':'rich_interpolation',
                                            'freq':'15min',
                                            'vision_concatenation_early' : True,
                                            'vision_concatenation_late' : False,
                                            'vision_grn_out_dim':32,
                                            'vision_num_heads':3,
                                            },
                    'large_attention_DA_rich_interpolation_nh2_emb64':{'dataset_names':['subway_in','subway_out'],
                                            'data_augmentation': True,
                                            'DA_method':'rich_interpolation',
                                            'freq':'15min',
                                            'vision_concatenation_early' : True,
                                            'vision_concatenation_late' : False,
                                            'vision_grn_out_dim':64,
                                            'vision_num_heads':2,
                                            },
                    'large_attention_DA_rich_interpolation_nh2_emb32':{'dataset_names':['subway_in','subway_out'],
                                            'data_augmentation': True,
                                            'DA_method':'rich_interpolation',
                                            'freq':'15min',
                                            'vision_concatenation_early' : True,
                                            'vision_concatenation_late' : False,
                                            'vision_grn_out_dim':32,
                                            'vision_num_heads':2,
                                            },
                    'large_attention_DA_rich_interpolation_nh2_emb16':{'dataset_names':['subway_in','subway_out'],
                                            'data_augmentation': True,
                                            'DA_method':'rich_interpolation',
                                            'freq':'15min',
                                            'vision_concatenation_early' : True,
                                            'vision_concatenation_late' : False,
                                            'vision_grn_out_dim':16,
                                            'vision_num_heads':2,
                                            },
                }

                        
    for add_name_id,config_diff in config_diffs.items():
        config_diff.update(modification)
        train_model_on_k_fold_validation(trial_id,load_config =True,
                                            save_folder=save_folder,
                                            modification=config_diff,
                                            add_name_id=add_name_id)


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



if False: 

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