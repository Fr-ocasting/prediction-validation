
# GET PARAMETERS
import os 
import sys
import torch 
import itertools
import pandas as pd 
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from examples.train_model_on_k_fold_validation import train_model_on_k_fold_validation,load_configuration,train_valid_1_model

if True: 
    save_folder = 'K_fold_validation/training_with_HP_tuning/re_validation'
    trial_id = 'subway_in_subway_out_STGCN_MSELoss_2025_02_19_00_05_19271'
    epochs_validation = 100
    args,folds = load_configuration(trial_id,True)

    modification ={'keep_best_weights':True,
                    'epochs':epochs_validation,
                    'device':torch.device("cuda:0"),
                    }
    
    L_epsilon = ['station_epsilon100','station_epsilon300']
    L_Apps = ['Google_Maps','Instagram','Deezer']

    config_diffs = {}
    Combination_Apps,CombinationTags = [],[]
    for i in range(len(L_Apps)):
        Combination_Apps = Combination_Apps+[list(x) for x in itertools.combinations(L_Apps,i+1)]
    for i in range(len(L_epsilon)):
        CombinationTags = CombinationTags +[list(x) for x in itertools.combinations(L_epsilon,i+1)]

    for NetMob_selected_apps,NetMob_selected_tags in list(itertools.product(Combination_Apps,CombinationTags)):
        name_config = f"NETMOB_eps{'_'.join([x.split('epsilon')[-1] for x in NetMob_selected_tags])}_{'_'.join(NetMob_selected_apps)}"
        config_diffs.update({name_config:{'dataset_names':['subway_in','netmob_POIs'],
                                            'data_augmentation': False,
                                            'freq':'15min',
                                            'NetMob_selected_apps':  NetMob_selected_apps,
                                            'NetMob_transfer_mode' :  ['DL'],
                                            'NetMob_selected_tags': NetMob_selected_tags,
                                            'NetMob_expanded' : '',
                                            'stacked_contextual': True,
                                            'temporal_graph_transformer_encoder': False,
                                            'compute_node_attr_with_attn': False,
                                            }
                                })
                        

    df_metrics_per_config = pd.DataFrame()
    for add_name_id,config_diff_i in config_diffs.items():
        config_diff_i.update(modification)
        trainer,args,training_mode_list,metric_list = train_valid_1_model(args,trial_id,save_folder,modification=config_diff_i)

        # Keep track on metrics :
        df_metrics_per_config.index = [f'{training_mode}_{metric}' for training_mode in training_mode_list for metric in metric_list]
        df_metrics_per_config[add_name_id] = [trainer.performance[f'{training_mode}_metrics'][metric] for training_mode in training_mode_list for metric in metric_list]

        df_metrics_per_config.to_csv('../save/results/NetMob_as_Channel.csv')



if False: 
    save_folder = 'K_fold_validation/training_with_HP_tuning/re_validation'
    trial_id = 'subway_in_subway_out_STGCN_VariableSelectionNetwork_MSELoss_2025_01_20_05_38_87836'
    epochs_validation = 100
    args,folds = load_configuration(trial_id,True)

    modification ={'keep_best_weights':True,
                    'epochs':epochs_validation,
                    'device':torch.device("cuda:0"),
                    }

    config_diffs = {'large_attention_DA_rich_interpolation_nh3_emb48':{'dataset_names':['subway_in','subway_out'],
                                            'data_augmentation': True,
                                            'DA_method':'rich_interpolation',
                                            'freq':'15min',
                                            'vision_concatenation_early' : True,
                                            'vision_concatenation_late' : False,
                                            'vision_grn_out_dim':48,
                                            'vision_num_heads':3,
                                            #'NetMob_selected_apps':  ['Google_Maps','Instagram','Deezer'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                                            #'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                            #'NetMob_selected_tags' : ['iris','station'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                            #'NetMob_expanded' : '_expanded', # '' # '_expanded'
                                            #'data_augmentation': True,
                                            #'DA_method':'magnitude_warping',
                                            },
                    'large_attention_DA_rich_interpolation_nh3_emb24':{'dataset_names':['subway_in','subway_out'],
                                            'data_augmentation': True,
                                            'DA_method':'rich_interpolation',
                                            'freq':'15min',
                                            'vision_concatenation_early' : True,
                                            'vision_concatenation_late' : False,
                                            'vision_grn_out_dim':24,
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