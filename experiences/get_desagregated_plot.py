import os 
import sys
import torch
import numpy as np
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from pipeline.Evaluation.STDA2.accuracy_comparison import get_model_args,get_desagregated_comparison_plot
from experiences.common_results import find_baseline
from experiences.common_parameter import convertion_exp_name



def get_desagregated_gains(dic_exp_to_names,
                           dic_trials,
                           horizons,
                           comparison_on_rainy_events,
                           range_k,
                           station_clustering,
                           folder_path,
                           save_bool = True,
                           heatmap= False,
                           daily_profile= False,
                           dendrogram= False,
                           dataset_names = None,
                           bool_plot = True,
                           clusters = None,
                           list_top_k_percent = None,
                           ):
    issue_while_loading_saved_weights = ''
    dic_bd_metrics_all = {}
    init_folder_path = f"{folder_path}/plot"  if folder_path is not None else None
    for exp_i,target_model_name in dic_exp_to_names.items():
        dic_bd_metrics_all[exp_i] = {}
        target_data = '_'.join(target_model_name.split('_')[:-1])
        model_name = target_model_name.split('_')[-1]
        configs = dic_trials[exp_i]

        if dataset_names is not None:
            exp_tmp = convertion_exp_name(target_data,dataset_names)
        else:
            exp_tmp = exp_i

        for h in horizons:
            print('   Horizon: ',h)
            dic_bd_metrics_all[exp_i][h] = {}
            baseline = find_baseline(exp_i,h=h,exp_tmp=exp_tmp,configs=configs)
            if (baseline in configs) and baseline.endswith(f"_h{h}"):
                print('   Baseline: ',baseline)
                trial_ids2 = [f"{config}_bis" for config in configs if (config.endswith(f"_h{h}")) and not(baseline == config)]
                trial_ids1 = [f"{baseline}_bis"]*len(trial_ids2)

                if exp_tmp == 'Exp4_15min_h1':
                    exp_tmp = 'Exp4_15min'
                save_folder_name = f'{exp_i}/{target_data}_{model_name}'
                save_folder_name_bis = None

                model_args,model_args_bis,path_model_args,path_model_args_bis = get_model_args(save_folder_name,save_folder_name_bis)

                print('Trial id Ref: ',trial_ids1[0])
                print('Trial id to test: ')
                for trial_id in trial_ids2:
                    print(trial_id)

                for trial_id1,trial_id2 in zip(trial_ids1,trial_ids2):
                    folder_path_i = f"{init_folder_path}/{exp_i}"
                    
                    if save_bool:
                        save_name = f"desag_{trial_id2}"
                    else:
                        save_name = None
                    outputs = get_desagregated_comparison_plot(
                                trial_id1,
                                trial_id2,
                                model_args = model_args,
                                model_args_bis = model_args_bis,
                                path_model_args = path_model_args,
                                path_model_args_bis = path_model_args_bis,
                                range_k = range_k,
                                trial_id1_in_bis=False,
                                trial_id2_in_bis=False,
                                comparison_on_rainy_events = comparison_on_rainy_events,
                                station_clustering = station_clustering,
                                folder_path = folder_path_i,
                                save_name = save_name,
                                heatmap = heatmap,
                                daily_profile = daily_profile,
                                dendrogram = dendrogram,
                                bool_plot = bool_plot,
                                clusters = clusters,
                                list_top_k_percent = list_top_k_percent,
                            )
                    clusterer,full_predict1,full_predict2,train_input,X,Y_true,L_trainers_1,L_trainers_2,ds1,ds2,args_init1,args_init2, dic_bd_metrics = outputs

                    # ====== SAVE ALL TOP K METRICS  ====
                    dic_bd_metrics_all[exp_i][h][trial_id1] = dic_bd_metrics[trial_id1]
                    dic_bd_metrics_all[exp_i][h][trial_id2] = dic_bd_metrics[trial_id2]


                    print('   Keys within dic_bd_metrics:\n',dic_bd_metrics.keys())


                    if full_predict1 is None:
                        issue_while_loading_saved_weights +=  f"\nProblem for {trial_id1} vs {trial_id2} in {exp_i}"
                    print(issue_while_loading_saved_weights)

            else:
                issue_while_loading_saved_weights +=  f"\nBaseline {baseline} not found in configs\n{configs}\nfor horizon {h} in {exp_i}\n or does not end with _h{h}"
                print(issue_while_loading_saved_weights)
    return dic_bd_metrics_all
            


if __name__ == "__main__":

    from experiences.common_results import dic_exp_to_names,dic_trials

    trial_id1_in_bis = False
    trial_id2_in_bis = False
    range_k = range(1,6) # range(1,6)
    comparison_on_rainy_events =  True #False 
    station_clustering = True 
    horizons = [4] # [1,4]
    init_folder_path = f"/home/rrochas/prediction-validation/save"

    # ---- For producing only Bike-out on rainy plots: --- 
    dic_exp_to_names = {'Exp2': dic_exp_to_names['Exp2']}


    print('dic_exp_to_names: ',dic_exp_to_names)
    print('dic_trials: ',dic_trials)
    print('\n--------------------------------')

    # ---- If folder does not exist in save:
    for exp_i in dic_exp_to_names.keys():
        if (exp_i == 'Exp1_subway_out'):
            folder_path = f"{init_folder_path}/plot/{exp_i}"
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
    
    get_desagregated_gains(dic_exp_to_names,dic_trials,horizons,comparison_on_rainy_events,range_k,station_clustering,init_folder_path,save_bool = True)
