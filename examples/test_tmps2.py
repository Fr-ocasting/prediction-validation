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
if True:
    save_folder = 'K_fold_validation/training_with_HP_tuning/re_validation'
    trial_id = 'subway_in_subway_out_STGCN_MSELoss_2025_02_19_00_05_19271'
    epochs_validation = 100
    args,folds = load_configuration(trial_id,True)
    modification ={'keep_best_weights':True,
                    'epochs':epochs_validation,
                    'device':torch.device("cuda:1"),
                    }
    

    L_epsilon = ['station_epsilon300'] #  ['station_epsilon100','station_epsilon300']
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
        name_config = f"NETMOB_eps{'_'.join([x.split('epsilon')[-1] for x in NetMob_selected_tags])}_{'_'.join(NetMob_selected_apps)}_1"
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


if False: 
    save_folder = 'K_fold_validation/training_with_HP_tuning/re_validation'
    trial_id = 'subway_in_subway_out_STGCN_MSELoss_2025_02_19_00_05_19271'
    epochs_validation = 100
    args,folds = load_configuration(trial_id,True)

    modification ={'keep_best_weights':True,
                    'epochs':epochs_validation,

                    'data_augmentation': True,
                    'DA_method': 'rich_interpolation',
                    'freq':'15min',

                    'learnable_adj_matrix' : False,
                    'stacked_contextual': True,

                    'temporal_graph_transformer_encoder': False,
                    'compute_node_attr_with_attn': False,
                    #'vision_concatenation_early' : True,
                    #'vision_concatenation_late' : False,
                    #'vision_model_name': 'VariableSelectionNetwork',

                    'device':torch.device("cuda:0"),
                    }
    
    config_diffs = {}
    config_diffs.update({'STANDARDIZE_subway_out_2':{'dataset_names':['subway_in','subway_out'],
                                                    'standardize': True,
                                                    'minmaxnorm':False,},
                          'STANDARDIZE_deezer_2': {'dataset_names':['subway_in','netmob_POIs'],
                                                'NetMob_only_epsilon': True,    # True # False
                                                'NetMob_selected_apps':  ['Deezer'],
                                                'NetMob_transfer_mode' :  ['DL'],
                                                'NetMob_selected_tags': ['station_epsilon100'],
                                                'NetMob_expanded' : '',
                                                'standardize': True,
                                                'minmaxnorm':False,
                                                },
                         'STANDARDIZE_Web_Ads_2': {'dataset_names':['subway_in','netmob_POIs'],
                                                'NetMob_only_epsilon': True,    # True # False
                                                'NetMob_selected_apps':  ['Web_Ads'],
                                                'NetMob_transfer_mode' :  ['DL'],
                                                'NetMob_selected_tags': ['station_epsilon100'],
                                                'NetMob_expanded' : '',
                                                'standardize': True,
                                                'minmaxnorm':False,
                                                },
                        'MINMAXNORM_subway_out_1':{'dataset_names':['subway_in','subway_out'],
                                                                            'standardize': False,
                                                                            'minmaxnorm':True,},
                        'MINMAXNORM_subway_out_2':{'dataset_names':['subway_in','subway_out'],
                                                                            'standardize': False,
                                                                            'minmaxnorm':True,},
                         'MINMAXNORM_deezer_1': {'dataset_names':['subway_in','netmob_POIs'],
                                                'NetMob_only_epsilon': True,    # True # False
                                                'NetMob_selected_apps':  ['Deezer'],
                                                'NetMob_transfer_mode' :  ['DL'],
                                                'NetMob_selected_tags': ['station_epsilon100'],
                                                'NetMob_expanded' : '',
                                                'standardize': False,
                                                'minmaxnorm':True,
                                                },
                         'MINMAXNORM_deezer_2': {'dataset_names':['subway_in','netmob_POIs'],
                                                'NetMob_only_epsilon': True,    # True # False
                                                'NetMob_selected_apps':  ['Deezer'],
                                                'NetMob_transfer_mode' :  ['DL'],
                                                'NetMob_selected_tags': ['station_epsilon100'],
                                                'NetMob_expanded' : '',
                                                'standardize': False,
                                                'minmaxnorm':True,
                                                },
                         'MINMAXNORM_Web_Ads_1': {'dataset_names':['subway_in','netmob_POIs'],
                                                'NetMob_only_epsilon': True,    # True # False
                                                'NetMob_selected_apps':  ['Web_Ads'],
                                                'NetMob_transfer_mode' :  ['DL'],
                                                'NetMob_selected_tags': ['station_epsilon100'],
                                                'NetMob_expanded' : '',
                                                'standardize': False,
                                                'minmaxnorm':True,
                                                },
                         'MINMAXNORM_Web_Ads_2': {'dataset_names':['subway_in','netmob_POIs'],
                                                'NetMob_only_epsilon': True,    # True # False
                                                'NetMob_selected_apps':  ['Web_Ads'],
                                                'NetMob_transfer_mode' :  ['DL'],
                                                'NetMob_selected_tags': ['station_epsilon100'],
                                                'NetMob_expanded' : '',
                                                'standardize': False,
                                                'minmaxnorm':True,
                                                },
                        })
    df_metrics_per_config = pd.DataFrame()
    for add_name_id,config_diff_i in config_diffs.items():
        config_diff_i.update(modification)
        train_model_on_k_fold_validation(trial_id,load_config =True,
                                save_folder=save_folder,
                                modification=config_diff_i,
                                add_name_id=add_name_id)
    
    if False:
        df_metrics_per_config = pd.DataFrame()
        for add_name_id,config_diff_i in config_diffs.items():
            config_diff_i.update(modification)
            trainer,args,training_mode_list,metric_list = train_valid_1_model(args,trial_id,save_folder,modification=config_diff_i)


if False: 
    save_folder = 'K_fold_validation/training_with_HP_tuning/re_validation'
    trial_id = 'subway_in_subway_out_STGCN_MSELoss_2025_02_19_00_05_19271'
    epochs_validation = 1
    args,folds = load_configuration(trial_id,True)

    modification ={'keep_best_weights':True,
                    'epochs':epochs_validation,
                    'temporal_graph_transformer_encoder': True,
                    'TGE_num_layers' : 4, #2
                    'TGE_num_heads' :  1, #IMPOSSIBLE > 1 CAR DOIT DIVISER L = 7
                    'TGE_FC_hdim' :  32, #32
                    'dropout': 0.271795,
                    'weight_decay': 0.018890,
                    'scheduler': True,
                    'torch_scheduler_milestone':28.0,
                    'torch_scheduler_gamma': 0.995835,
                    'torch_scheduler_lr_start_factor': 0.880994,
                    'device':torch.device("cuda:1"),
                    }

    config_diffs = {'RE_TGE_l4_h1_fc32_lr00100':{'lr': 0.00100},
                    'RE_TGE_l4_h1_fc32_lr0.00050':{'lr': 0.0005},
                    'RE_TGE_l4_h1_fc32_lr0.00025':{'lr': 0.00025},
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
                    'device':torch.device("cuda:1"),
                    }

    config_diffs = {'large_attention':{'dataset_names':['subway_in','netmob_POIs'],
                                            'data_augmentation': True,
                                            'DA_method':'rich_interpolation',
                                            'freq':'15min',
                                            'vision_concatenation_early' : True,
                                            'vision_concatenation_late' : False,
                                            'NetMob_selected_apps':  ['Google_Maps','Instagram','Deezer'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                                            'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                            'NetMob_selected_tags' : ['iris','station'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                            'NetMob_expanded' : '_expanded', # '' # '_expanded'
                                            #'data_augmentation': True,
                                            #'DA_method':'magnitude_warping',
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
                    'device':torch.device("cuda:1"),
                    }

    config_diffs = {'RE_CRITER_3lanes_netmob_POIS_rich_interpolation_Waze_DL':{'dataset_names':['CRITER_3lanes','netmob_POIs'],
                                                                'data_augmentation': True,
                                                                'DA_method':'rich_interpolation',
                                                                'freq':'30min',
                                                                'vision_concatenation_early' : True,
                                                                'vision_concatenation_late' : False,
                                                                'NetMob_selected_apps':  ['Waze'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                                                                'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                'NetMob_expanded' : '', # '' # '_expanded'
                                                                },
                    'RE_CRITER_3lanes_netmob_POIS_rich_interpolation_Waze_Deezer_Maps_DL':{'dataset_names':['CRITER_3lanes','netmob_POIs'],
                                                                'data_augmentation': True,
                                                                'DA_method':'rich_interpolation',
                                                                'freq':'30min',
                                                                'vision_concatenation_early' : True,
                                                                'vision_concatenation_late' : False,
                                                                'NetMob_selected_apps':  ['Waze','Deezer','Google_Maps'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                                                                'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                'NetMob_expanded' : '', # '' # '_expanded'
                                                                },
                    'RE_CRITER_3lanes_netmob_POIS_rich_interpolation_Waze_Deezer_Maps_DL_UL':{'dataset_names':['CRITER_3lanes','netmob_POIs'],
                                                                'data_augmentation': True,
                                                                'DA_method':'rich_interpolation',
                                                                'freq':'30min',
                                                                'vision_concatenation_early' : True,
                                                                'vision_concatenation_late' : False,
                                                                'NetMob_selected_apps':  ['Waze','Deezer','Google_Maps'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                                                                'NetMob_transfer_mode' :  ['DL','UL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                'NetMob_expanded' : '', # '' # '_expanded'
                                                                },
                    'RE_CRITER_3lanes_netmob_POIS_rich_interpolation_Waze_DL_nh8_emb64':{'dataset_names':['CRITER_3lanes','netmob_POIs'],
                                                                        'data_augmentation': True,
                                                                        'DA_method':'rich_interpolation',
                                                                        'freq':'30min',
                                                                        'vision_concatenation_early' : True,
                                                                        'vision_concatenation_late' : False,
                                                                        'vision_grn_out_dim':64,
                                                                        'vision_num_heads':8,
                                                                        'NetMob_selected_apps':  ['Waze'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                                                                        'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                        'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                        'NetMob_expanded' : '', # '' # '_expanded'
                                                                        
                                                                        },
                    'RE_CRITER_3lanes_netmob_POIS_rich_interpolation_Waze_Deezer_Maps_DL_nh8_emb128':{'dataset_names':['CRITER_3lanes','netmob_POIs'],
                                                                'data_augmentation': True,
                                                                'DA_method':'rich_interpolation',
                                                                'freq':'30min',
                                                                'vision_concatenation_early' : True,
                                                                'vision_concatenation_late' : False,
                                                                'vision_grn_out_dim':128,
                                                                'vision_num_heads':8,
                                                                'NetMob_selected_apps':  ['Waze','Deezer','Google_Maps'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                                                                'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                'NetMob_expanded' : '', # '' # '_expanded'
                                                                },
                    'RE_CRITER_3lanes_netmob_POIS_rich_interpolation_Waze_Deezer_Maps_DL_UL_nh8_emb128':{'dataset_names':['CRITER_3lanes','netmob_POIs'],
                                                                'data_augmentation': True,
                                                                'DA_method':'rich_interpolation',
                                                                'freq':'30min',
                                                                'vision_concatenation_early' : True,
                                                                'vision_concatenation_late' : False,
                                                                'vision_grn_out_dim':128,
                                                                'vision_num_heads':8,
                                                                'NetMob_selected_apps':  ['Waze','Deezer','Google_Maps'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                                                                'NetMob_transfer_mode' :  ['DL','UL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                                'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                                'NetMob_expanded' : '', # '' # '_expanded'
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
                    'device':torch.device("cuda:1"),
                    }

    config_diffs = {'RE_CRITER_3lanes_rich_interpolation':{'dataset_names':['CRITER_3lanes'],
                                                                'data_augmentation': True,
                                                                'DA_method':'rich_interpolation',
                                                                'freq':'30min'},
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
    epochs_validation = 1
    args,folds = load_configuration(trial_id,True)

    modification ={'keep_best_weights':True,
                    'epochs':epochs_validation,
                    'device':torch.device("cuda:1"),
                    'loss_function_type':'quantile',
                    'alpha':0.05,
                    'track_pi':True,
                    'type_calib':'classic',
                    }

    config_diffs = {'USELESS':{'dataset_names':['subway_in'],
                                            'data_augmentation': False,
                                            'freq':'15min'
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
    epochs_validation = 1
    args,folds = load_configuration(trial_id,True)

    modification ={'keep_best_weights':True,
                    'epochs':epochs_validation,
                    'device':torch.device("cuda:1"),
                    'loss_function_type':'quantile',
                    'alpha':0.05,
                    'track_pi':True,
                    'type_calib':'classic',
                    }



    config_diffs = {'USELESS':{'dataset_names':['subway_in','netmob_POIs_per_station'],
                                            'vision_concatenation_early' : True,
                                            'vision_concatenation_late' : False,
                                            'data_augmentation': False,
                                            'freq':'30min',
                                            'NetMob_selected_apps':  ['Google_Maps','Deezer','Instagram'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                                            'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                            'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                            'NetMob_expanded' : '', # '' # '_expanded'
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
                    'device':torch.device("cuda:1"),
                    }


    config_diffs = {'rich_interpolation_identical':{'dataset_names':['subway_in'],
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



if False : 
    save_folder = 'K_fold_validation/training_with_HP_tuning/re_validation'

    trial_id = 'subway_in_subway_out_STGCN_VariableSelectionNetwork_MSELoss_2025_01_20_05_38_87836'
    epochs_validation = 100
    args,folds = load_configuration(trial_id,True)

    modification ={'keep_best_weights':True,
                    'epochs':epochs_validation,
                    'device':torch.device("cuda:1"),
                    }

    config_diffs = {'NETMOB_POIS_DA_magnitude_maps_deezer_insta_DL_IRIS':{'dataset_names':['subway_in','netmob_POIs'],
                                        'vision_concatenation_early' : True,
                                        'vision_concatenation_late' : False,
                                        'NetMob_selected_apps':  ['Google_Maps','Deezer','Instagram'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                                        'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                        'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                        'NetMob_expanded' : '', # '' # '_expanded'
                                        'data_augmentation': True,
                                        'DA_method':'magnitude_warping',
                                        },
                    'NETMOB_POIS_DA_magnitude_rich_interpolation_maps_deezer_insta_DL_IRIS':{'dataset_names':['subway_in','netmob_POIs'],
                                        'vision_concatenation_early' : True,
                                        'vision_concatenation_late' : False,
                                        'NetMob_selected_apps':  ['Google_Maps','Deezer','Instagram'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                                        'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                        'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                        'NetMob_expanded' : '', # '' # '_expanded'
                                        'data_augmentation': True,
                                        'DA_method':['rich_interpolation','magnitude_warping'],
                                        },
                    'NETMOB_POIS_DA_magnitude_rich_interpolation_maps_deezer_insta_DL_IRIS_station_expanded':{'dataset_names':['subway_in','netmob_POIs'],
                                        'vision_concatenation_early' : True,
                                        'vision_concatenation_late' : False,
                                        'NetMob_selected_apps':  ['Google_Maps','Deezer','Instagram'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                                        'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                        'NetMob_selected_tags' : ['iris','station'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                        'NetMob_expanded' : '_expanded', # '' # '_expanded'
                                        'data_augmentation': True,
                                        'DA_method':['rich_interpolation','magnitude_warping'],
                                        },
                   'NETMOB_POIS_DA_magnitude_rich_interpolation_maps_deezer_insta_DL_IRIS_station_stadium_uni':{'dataset_names':['subway_in','netmob_POIs'],
                                        'vision_concatenation_early' : True,
                                        'vision_concatenation_late' : False,
                                        'NetMob_selected_apps':  ['Google_Maps','Deezer','Instagram'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                                        'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                        'NetMob_selected_tags' : ['iris','station','stadium','university'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                        'NetMob_expanded' : '_expanded', # '' # '_expanded'
                                        'data_augmentation': True,
                                        'DA_method':['rich_interpolation','magnitude_warping'],
                                        }
                }

                    
    for add_name_id,config_diff in config_diffs.items():
        config_diff.update(modification)
        train_model_on_k_fold_validation(trial_id,load_config =True,
                                            save_folder=save_folder,
                                            modification=config_diff,
                                            add_name_id=add_name_id)
