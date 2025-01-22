
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


save_folder = 'K_fold_validation/training_with_HP_tuning/re_validation'

trial_id = 'subway_in_subway_out_STGCN_VariableSelectionNetwork_MSELoss_2025_01_20_05_38_87836'
epochs_validation = 100
args,folds = load_configuration(trial_id,True)

modification ={'keep_best_weights':True,
                'epochs':epochs_validation,
                'device':torch.device("cuda:1"),
                }



config_diffs = {'NETMOB_POIS_maps_deezer_insta_Twitter_concat_early':{'dataset_names':['subway_in','netmob_POIs'],
                                                'vision_concatenation_early' : True,
                                                'vision_concatenation_late' : False,
                                                'NetMob_selected_apps':  ['Google_Maps','Deezer','Instagram','Twitter'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                                                'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                'NetMob_selected_tags' : ['station','iris','stadium','university'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                'NetMob_expanded' : '' # '' # '_expanded'
                },
                'NETMOB_POIS_maps_deezer_insta_Twitter_concat_early_UL_DL':{'dataset_names':['subway_in','netmob_POIs'],
                                                'vision_concatenation_early' : True,
                                                'vision_concatenation_late' : False,
                                                'NetMob_selected_apps':  ['Google_Maps','Deezer','Instagram','Twitter'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                                                'NetMob_transfer_mode' :  ['DL','UL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                'NetMob_selected_tags' : ['station','iris','stadium','university'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                'NetMob_expanded' : '' # '' # '_expanded'
                },
                'NETMOB_POIS_maps_deezer_insta_Twitter_concat_early_late_UL_DL':{'dataset_names':['subway_in','netmob_POIs'],
                                                'vision_concatenation_early' : True,
                                                'vision_concatenation_late' : True,
                                                'NetMob_selected_apps':  ['Google_Maps','Deezer','Instagram','Twitter'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                                                'NetMob_transfer_mode' :  ['DL','UL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                'NetMob_selected_tags' : ['station','iris','stadium','university'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                'NetMob_expanded' : '' # '' # '_expanded'
                }
            }


                    
for add_name_id,config_diff in config_diffs.items():
    config_diff.update(modification)
    train_model_on_k_fold_validation(trial_id,load_config =True,
                                        save_folder=save_folder,
                                        modification=config_diff,
                                        add_name_id=add_name_id)