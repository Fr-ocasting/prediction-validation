import os 
import sys
import torch
import numpy as np
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from examples.accuracy_comparison import get_model_args,get_desagregated_comparison_plot,get_previous
from experiences.common_results import find_baseline
from pipeline.utils.metrics import evaluate_metrics

def get_desagregated_gains(dic_exp_to_names,dic_trials,horizons,comparison_on_rainy_events,range_k,station_clustering,folder_path,save_bool = True,heatmap= False,daily_profile= False,dendrogram= False):
    issue_while_loading_saved_weights,log = '', ''
    init_folder_path = f"{folder_path}/plot"
    for exp_i,target_model_name in dic_exp_to_names.items():
        target_data = '_'.join(target_model_name.split('_')[:-1])
        model_name = target_model_name.split('_')[-1]
        print(exp_i)
        configs = dic_trials[exp_i]
        for h in horizons:
            print('   Horizon: ',h)
            baseline = find_baseline(exp_i,h=h)
            if (baseline in configs) and baseline.endswith(f"_h{h}"):
                print('   Baseline: ',baseline)
                trial_ids2 = [f"{config}_bis" for config in configs if (config.endswith(f"_h{h}")) and not(baseline == config)]
                trial_ids1 = [f"{baseline}_bis"]*len(trial_ids2)

                if exp_i == 'Exp4_15min_h1':
                    exp_i = 'Exp4_15min'
                save_folder_name = f'{exp_i}/{target_data}_{model_name}'
                save_folder_name_bis = None

                model_args,model_args_bis,path_model_args,path_model_args_bis = get_model_args(save_folder_name,save_folder_name_bis)

                print('Trial id Ref: ',trial_ids1[0])
                print('Trial id to test: ')
                for trial_id in trial_ids2:
                    print(trial_id)

                for trial_id1,trial_id2 in zip(trial_ids1,trial_ids2):
                    if comparison_on_rainy_events:
                        folder_path = f"{init_folder_path}/{exp_i}_rainy"
                    else:
                        folder_path = f"{init_folder_path}/{exp_i}"
                    if save_bool:
                        save_name = f"desag_{trial_id2}"
                    else:
                        save_name = None
                    outputs = get_desagregated_comparison_plot(trial_id1,trial_id2,
                                                                model_args = model_args,
                                                                model_args_bis = model_args_bis,
                                                                path_model_args = path_model_args,
                                                                path_model_args_bis = path_model_args_bis,
                                                                range_k = range_k,
                                                                trial_id1_in_bis=False,
                                                                trial_id2_in_bis=False,
                                                                comparison_on_rainy_events = comparison_on_rainy_events ,
                                                                station_clustering = station_clustering,
                                                                folder_path = folder_path,
                                                                save_name = save_name,
                                                                heatmap = heatmap,
                                                                daily_profile = daily_profile,
                                                                dendrogram = dendrogram

                                                                )
                    clusterer,full_predict1,full_predict2,train_input,X,Y_true,L_trainers_1,L_trainers_2,ds1,ds2,args_init1,args_init2,rainy_indices,rainy_mask = outputs

                    if full_predict1 is None:
                        issue_while_loading_saved_weights +=  f"\nProblem for {trial_id1} vs {trial_id2} in {exp_i}"
                    print(issue_while_loading_saved_weights)

                    h_idx = args_init2.step_ahead // args_init2.horizon_step
                    previous = get_previous(X,Y_true,h_idx)

                    if comparison_on_rainy_events :
                        full_predict1 = torch.index_select(full_predict1,0,rainy_indices)
                        full_predict2 = torch.index_select(full_predict2,0,rainy_indices)
                        Y_true = torch.index_select(Y_true,0,rainy_indices)
                        previous = torch.index_select(previous,0,rainy_indices)

                    RMSE1,MAE1,MASE1,MAPE1 = [],[],[],[]
                    RMSE2,MAE2,MASE2,MAPE2 = [],[],[],[]
                    for n_bis in range(full_predict1.size(-1)):
                        full_predict1_i = full_predict1[...,n_bis] 
                        full_predict2_i = full_predict2[...,n_bis] 
                        dic_metric1_i = evaluate_metrics(full_predict1_i,Y_true,metrics = ['rmse','mse','mae','mase','mape'], previous = previous,horizon_step = h_idx)
                        dic_metric2_i = evaluate_metrics(full_predict2_i,Y_true,metrics = ['rmse','mse','mae','mase','mape'], previous = previous,horizon_step = h_idx)
                        RMSE1.append(dic_metric1_i['rmse_all'])
                        MAE1.append(dic_metric1_i['mae_all'])
                        MASE1.append(dic_metric1_i['mase_all'])
                        MAPE1.append(dic_metric1_i['mape_all'])
                        RMSE2.append(dic_metric2_i['rmse_all'])
                        MAE2.append(dic_metric2_i['mae_all'])
                        MASE2.append(dic_metric2_i['mase_all'])
                        MAPE2.append(dic_metric2_i['mape_all'])
                    RMSE1 = np.mean(np.array(RMSE1))
                    MAE1 = np.mean(np.array(MAE1))
                    MASE1 = np.mean(np.array(MASE1))
                    MAPE1 = np.mean(np.array(MAPE1))
                    RMSE2 = np.mean(np.array(RMSE2))
                    MAE2 = np.mean(np.array(MAE2))
                    MASE2 = np.mean(np.array(MASE2))
                    MAPE2 = np.mean(np.array(MAPE2))


                    if log == '':
                        log += f"{trial_id1[:-3]}:   All Steps RMSE = {RMSE1:.5f}, MAE = {MAE1:.5f}, MASE = {MASE1:.5f}, MAPE = {MAPE1:.5f}\n"
                    log += f"{trial_id2[:-3]}:   All Steps RMSE = {RMSE2:.5f}, MAE = {MAE2:.5f}, MASE = {MASE2:.5f}, MAPE = {MAPE2:.5f}\n"
                    print(log)
            else:
                issue_while_loading_saved_weights +=  f"\nBaseline {baseline} not found in configs\n{configs}\nfor horizon {h} in {exp_i}\n or does not end with _h{h}"
                print(issue_while_loading_saved_weights)
            


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
