
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


"""Evaluation de qualité des série temporelle NetMob de manière individuelle."""
if False:
    save_folder = 'K_fold_validation/training_with_HP_tuning/ASTGCN_2025_04_21_20_06_76371'
    #trial_id = 'subway_in_subway_out_STGCN_MSELoss_2025_02_19_00_05_19271'
    #trial_id = 'subway_in_subway_out_STGCN_MSELoss_2025_03_29_00_17_68381'
    trial_id = 'subway_in_subway_out_ASTGCN_MSELoss_2025_04_21_20_06_76371'
    epochs_validation = 160
    args,folds = load_configuration(trial_id,True)
    modification ={'keep_best_weights':True,
                    'epochs':epochs_validation,
                    'device':torch.device("cuda:0"),
                    #'standardize': False,
                    #'minmaxnorm':True,
                    #'learnable_adj_matrix' : False,
                    #'stacked_contextual': True,
                    #'temporal_graph_transformer_encoder': False,
                    #'compute_node_attr_with_attn': False,
                    #'freq': '15min',
                    }
    
    config_diffs = {'subway_in_subway_out':{}, 
                    'subway_in_subway_in':{'dataset_names':['subway_in','subway_in']},  
                    'station_epsilon100_Google_Maps':{'dataset_names':['subway_in','netmob_POIs'],
                                                    'NetMob_only_epsilon': True,   
                                                    'NetMob_selected_apps':  ['Google_Maps'],
                                                    'NetMob_transfer_mode' :  ['DL'],
                                                    'NetMob_selected_tags': ['station_epsilon100'],
                                                    'NetMob_expanded' : ''},
                    'station_epsilon100_Web_Weather':{'dataset_names':['subway_in','netmob_POIs'],
                                                    'NetMob_only_epsilon': True,   
                                                    'NetMob_selected_apps':  ['Web_Weather'],
                                                    'NetMob_transfer_mode' :  ['DL'],
                                                    'NetMob_selected_tags': ['station_epsilon100'],
                                                    'NetMob_expanded' : ''},
                    'station_epsilon100_Web_Downloads':{'dataset_names':['subway_in','netmob_POIs'],
                                                    'NetMob_only_epsilon': True,   
                                                    'NetMob_selected_apps':  ['Web_Downloads'],
                                                    'NetMob_transfer_mode' :  ['DL'],
                                                    'NetMob_selected_tags': ['station_epsilon100'],
                                                    'NetMob_expanded' : ''},
                    'station_epsilon100_Web_Deezer':{'dataset_names':['subway_in','netmob_POIs'],
                                                    'NetMob_only_epsilon': True,   
                                                    'NetMob_selected_apps':  ['Deezer'],
                                                    'NetMob_transfer_mode' :  ['DL'],
                                                    'NetMob_selected_tags': ['station_epsilon100'],
                                                    'NetMob_expanded' : ''}
                                                      }
                        

    for add_name_id,config_diff in config_diffs.items():
        config_diff.update(modification)
        train_model_on_k_fold_validation(trial_id,load_config =True,
                                            save_folder=save_folder,
                                            modification=config_diff,
                                            add_name_id=add_name_id)

"""Evaluation de qualité des série temporelle NetMob de manière individuelle."""
if False:
    save_folder = 'K_fold_validation/training_with_HP_tuning/re_validation_epsilon100'
    trial_id = 'subway_in_subway_out_STGCN_MSELoss_2025_02_19_00_05_19271'
    epochs_validation = 100
    args,folds = load_configuration(trial_id,True)
    modification ={'keep_best_weights':True,
                    'epochs':epochs_validation,
                    'device':torch.device("cuda:0"),
                    }
    

    L_epsilon = ['station_epsilon100'] #  ['station_epsilon100','station_epsilon300']
    L_Apps = ['Apple_Video','Google_Play_Store','Google_Maps','Web_Clothes','Uber', 'Twitter',
            'Microsoft_Mail', 'Microsoft_Store', 'Apple_Music', 'Microsoft_Office', 'Pokemon_GO', 'Clash_of_Clans', 'Yahoo_Mail', 'PlayStation',
            'Wikipedia', 'Apple_Web_Services', 'Pinterest', 'Web_Ads', 'Google_Mail', 'Google_Meet',
            'Apple_Siri', 'Web_Adult', 'Spotify', 'Deezer', 'Waze', 'Web_Games', 'Apple_App_Store', 'Microsoft_Skydrive', 'Google_Docs', 'Microsoft_Web_Services',
            'Molotov', 'YouTube', 'Apple_iTunes', 'Apple_iMessage', 'DailyMotion', 'Netflix', 'Web_Transportation',
            'Web_Downloads', 'SoundCloud', 'TeamViewer', 'Google_Web_Services', 'Facebook', 'EA_Games', 'Tor', 'Amazon_Web_Services',
            'Web_e-Commerce', 'Telegram', 'Apple_Mail','Dropbox', 'Web_Food', 'Apple_iCloud', 'Skype', 'Facebook_Messenger', 'Twitch', 'Microsoft_Azure',
            'Instagram', 'Facebook_Live', 'Web_Streaming', 'Orange_TV', 'Periscope', 'Snapchat' ,'Web_Finance' ,'WhatsApp', 'Web_Weather','Google_Drive','LinkedIn','Yahoo','Fortnite']


    config_diffs = {}
    Combination_Apps = [[app_i] for app_i in L_Apps]
    CombinationTags =  [[epsi_i] for epsi_i in L_epsilon]

    for NetMob_selected_apps,NetMob_selected_tags in list(itertools.product(Combination_Apps,CombinationTags)):
        name_config = f"NETMOB_eps{'_'.join([x.split('epsilon')[-1] for x in NetMob_selected_tags])}_{'_'.join(NetMob_selected_apps)}_3"
        config_diffs.update({name_config:{'dataset_names':['subway_in','netmob_POIs'],
                                            'data_augmentation': True,
                                            'DA_method': 'rich_interpolation',
                                            'freq':'15min',
                                            
                                            'NetMob_only_epsilon': True,    # True # False
                                            'NetMob_selected_apps':  NetMob_selected_apps,
                                            'NetMob_transfer_mode' :  ['DL'],
                                            'NetMob_selected_tags': NetMob_selected_tags,
                                            'NetMob_expanded' : '',

                                            'standardize': False,
                                            'minmaxnorm':True,

                                            'learnable_adj_matrix' : False,
                                            
                                            'stacked_contextual': True,
                                            'temporal_graph_transformer_encoder': False,
                                            'compute_node_attr_with_attn': False,
                                            }
                                })
                        

    df_metrics_per_config = pd.DataFrame()
    for add_name_id,config_diff_i in config_diffs.items():
        config_diff_i.update(modification)
        train_model_on_k_fold_validation(trial_id,load_config =True,
                                save_folder=save_folder,
                                modification=config_diff_i,
                                add_name_id=add_name_id)


if True: 
    save_folder = 'K_fold_validation/training_with_HP_tuning/subway_in_subway_out_STGCN_MSELoss_2025_05_02_15_47_82038'
    trial_id = 'subway_in_subway_out_STGCN_MSELoss_2025_05_02_15_47_82038'
    epochs_validation = 300
    args,folds = load_configuration(trial_id,True)
    modification = {'epochs':epochs_validation}
    
    config_diffs = {}
    config_diffs.update({'trial_1':{}
                         })
                        

    df_metrics_per_config = pd.DataFrame()
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