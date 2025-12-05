import os 
import sys
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from examples.accuracy_comparison import get_model_args,get_desagregated_comparison_plot
from experiences.common_results import dic_exp_to_names,dic_trials,find_baseline


issue_while_loading_saved_weights = ''

trial_id1_in_bis = False
trial_id2_in_bis = False
range_k = range(1,6) # range(1,6)
comparison_on_rainy_events = False 
station_clustering = True 

init_folder_path = f"/home/rrochas/prediction-validation/save/plot"

# ---- If folder does not exist in save:
for exp_i in dic_exp_to_names.keys():
    folder_path = f"{init_folder_path}/{exp_i}"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


for exp_i,target_model_name in dic_exp_to_names.items():
    target_data = '_'.join(target_model_name.split('_')[:-1])
    model_name = target_model_name.split('_')[-1]

    # if not (exp_i == 'Exp2'):
    #     continue

    if (exp_i == 'Exp2'):
        continue

    print(exp_i)
    configs = dic_trials[exp_i]
    for h in [1,4]:
        baseline = find_baseline(exp_i,h=h)
        if (baseline in configs) and baseline.endswith(f"_h{h}"):
            trial_ids2 = [f"{config}_bis" for config in configs if (config.endswith(f"_h{h}")) and not(baseline == config)]
            trial_ids1 = [f"{baseline}_bis"]*len(trial_ids2)

            if exp_i == 'Exp4_15min_h1':
                exp_i = 'Exp4_15min'
            save_folder_name = f'{exp_i}/{target_data}_{model_name}'
            save_folder_name_bis = None

            model_args,model_args_bis,path_model_args,path_model_args_bis = get_model_args(save_folder_name,save_folder_name_bis)

            # # Check if all trials are accessible: 
            # for trial_id1,trial_id2 in zip(trial_ids1,trial_ids2):
                # t1_id = f"{trial_id1}1_f5"
                # t2_id = f"{trial_id2}1_f5"
                # assert t1_id in model_args['model'].keys(), f"trial_id1 {t1_id} not in model_args"
                # assert t2_id in model_args['model'].keys(),f"trial_id2 {t2_id} not in model_args"

            print('Trial id Ref: ',trial_ids1[0])
            print('Trial id to test: ')
            for trial_id in trial_ids2:
                print(trial_id)

            for trial_id1,trial_id2 in zip(trial_ids1,trial_ids2):
                if comparison_on_rainy_events:
                    folder_path = f"{init_folder_path}/{exp_i}_rainy"
                else:
                    folder_path = f"{init_folder_path}/{exp_i}"
                save_name = f"desag_{trial_id2}"
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
                                                            )
                clusterer,full_predict1,full_predict2,train_input,Y_true,L_trainers_1,L_trainers_2,ds1,ds2,args_init1,args_init2 = outputs

                if full_predict1 is None:
                    issue_while_loading_saved_weights +=  f"\nProblem for {trial_id1} vs {trial_id2} in {exp_i}"
                print(issue_while_loading_saved_weights)