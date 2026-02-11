# ==================================================
# IMPORT
import re 
import os 
import pandas as pd
import sys
import torch 
import importlib
import itertools 
import copy 
import torch._dynamo as dynamo; dynamo.graph_break()
torch._dynamo.config.verbose=True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from pipeline.utils.loger import LOG
from experiences.common_parameter import possible_target_kwargs, weather_possible_contextual_kwargs, model_configurations
from experiences.common_parameter import REPEAT_TRIAL,netmob_preprocessing_kwargs
from experiences.get_desagregated_plot import get_desagregated_gains
from experiences.compilation_parameter import compilation_modification
from experiences.loop_train_save_log import loop_train_save_log
from experiences.pipeline_desag.build_config_single_contextual import ConfigBuilder
from experiences.pipeline_desag.build_baseline_config import BaselineConfigBuilder
from experiences.pipeline_desag.utils import plotting_boxplot_of_trials
from experiences.pipeline_desag.MetricExporter import MetricExporter
from constants.paths import ROOT
inside_saved_folder = 'K_fold_validation/training_wo_HP_tuning'
folder_path = f"{ROOT}/save/{inside_saved_folder}"

# ==================================================
# SET PARAMETERS OF THE EXPERIMENT AND LOGER: 
'''
- Do not set seed to have different initialization for each trial
- SANITY_CHECKER: If True, Keep track of the gradients and the weights during the training to detect possible problems.
'''
# ------------------   
# exp_i = 'pipeline_subway_in'
    # contextual:
        # []
        # ['subway_out']
    # dataset_for_coverage = ['subway_in','subway_out']

# ------------------
# exp_i = 'pipeline_subway_in_netmob '
    # contextual:
        # []
        # ['subway_out']
        # ['subway_out', 'netmob']
        # ['netmob']
    # dataset_for_coverage = ['subway_in','subway_out','netmob_POIs']

# ------------------
# exp_i = 'pipeline_subway_out_netmob'
    # contextual:
        # []
        # ['subway_in']
        # ['subway_in', 'netmob']
        # ['netmob']
    # dataset_for_coverage = ['subway_in','subway_out','netmob_POIs']
    
# ------------------
# exp_i = 'pipeline_bike_out' 
    # contextual:
        # []
        # ['bike_in'], 
        # ['bike_in','subway_in_subway_out'], 
        # ['subway_in_subway_out'],
    # dataset_for_coverage = ['bike_out','subway_in','subway_out']

# ------------------
# exp_i = 'pipeline_bike_out_netmob'
    # contextual:   
        # []
        # ['bike_in']
        # ['bike_in', 'netmob']
        # ['bike_in', 'subway_in_subway_out']
        # ['bike_in', 'subway_in_subway_out', 'netmob']   
        # ['netmob']
        # ['netmob', 'subway_in_subway_out']
        # ['subway_in_subway_out']
        # dataset_for_coverage = ['bike_out','subway_in','subway_out','netmob_POIs']

# ------------------------------------------
exp_i = 'pipeline_bike_out'
freq = '15min' #'15min'  
horizons = [1,4] # [4]  #[1,4]
model_name = 'STAEformer'
target_data = 'bike_out' 
dataset_for_coverage = ['bike_in','bike_out','subway_in','subway_out']
possible_contextual_dataset_names = [
                                    ['bike_in'],
                                    ['subway_in_subway_out'],
                                    # ['subway_in_subway_out','bike_in','weather'],
                                     ['subway_in_subway_out','bike_in'],
                                    #  ['weather'],
                                    #  ['weather','bike_in'],
                                    #  ['weather','subway_in_subway_out'],
                                     ] # ['netmob_POIs'] #['subway_out']
TRIVIAL_TEST = False
REPEAT_TRIAL = 1 #5 
list_top_k_percent = [0.2,-0.2,None,0.8] # [-0.2,None, 0.2, 0.8]
comparison_on_rainy_events = True
# ------------------------------------------

training_save_folder = f'{inside_saved_folder}/{exp_i}' # f'K_fold_validation/training_wo_HP_tuning/{exp_i}' 
save_path_figures = f'{current_file_path}/results/plot/{exp_i}'
device = torch.device('cuda:0')
add_name_save = '' #'_clipping'  # ''  # '_trial2'

station_clustering = True
for contextual_dataset_names in possible_contextual_dataset_names:
    # assert len(contextual_dataset_names) == 1, "Only one contextual dataset at a time is allowed for this pipeline. Otherwise, update 'build_config_single_contextual.py' accordingly. "

    loger = LOG()
    SANITY_CHECKER = False # True
    metrics = ['RMSE','MAE','MASE']
    config_backbone_model = model_configurations[model_name]
    config_backbone_model['batch_size'] = 128 # 150 #80
    config_backbone_model['epochs'] = 100
    compilation_modification['torch_compile'] = 'compile' #'compile' # 'compile' # 'compile'  # False 
    compilation_modification['device'] = device
    # compilation_modification['grad_clipping_norm'] = 1.0

    # -----If small run if needed: 
    if TRIVIAL_TEST:
        config_backbone_model['epochs'] = 1 # 150 # 150 #80
        config_backbone_model['batch_size'] = 256 # 150 #80
        REPEAT_TRIAL  = 1
        compilation_modification['torch_compile'] = False

    # ==================================================
    # ASSERT SAVE PATH EXISTS & NOT OVERWRITE:
    # --- if save plot does not existe, mkdir: 

    # --- if save training save path already exist, display error & do not overwrite & continue next training

    # --- if save training folder path does not existe, mkdir:


    if True:
        # ==================================================
        # LOAD CONFIGURATIONS TO TEST:
        possible_contextual_kwargs = {}
        # possible_contextual_kwargs[ds][fusion][feature_extractor]
        print('contextual_dataset_names:\n ',contextual_dataset_names)
        for dataset_name in contextual_dataset_names:
            if 'weather' == dataset_name:
                contextual_kwargs = weather_possible_contextual_kwargs['early_fusion']['repeat_t_proj']
                possible_contextual_kwargs[dataset_name] = {'early_fusion':{'repeat_t_proj':contextual_kwargs}}

            else:
                path = f"experiences.pipeline_desag.{target_data}_pred.{dataset_name}_contextual"
                module = importlib.import_module(path)
                importlib.reload(module)
                contextual_kwargs = module.get_possible_contextual_kwarg(add_name_save)     
                possible_contextual_kwargs[dataset_name] = contextual_kwargs
        

        print('\n------------------------------- CONFIGURATIONS ----------------------------------\n')
        print(possible_contextual_kwargs)
        for predictive_data in possible_contextual_kwargs.keys():
            print(f'±n--- Predictive data: {predictive_data} ---')
            for fusion_type in possible_contextual_kwargs[predictive_data].keys():
                print(f'  -- Fusion type: {fusion_type} ---')
                for trial_id in possible_contextual_kwargs[predictive_data][fusion_type].keys():
                    print(f'    {trial_id}')
        print('\n---------------------------------------------------------------------------------\n')
        # ==================================================
        # LOAD CONFIG DICTIONARY: 
        configbuilder = ConfigBuilder(target_data,contextual_dataset_names,dataset_for_coverage,model_name,horizons,freq,REPEAT_TRIAL,SANITY_CHECKER,compilation_modification)
        dic_configs = configbuilder.build_config_single_contextual(
                                                    dic_configs = {},
                                                    possible_target_kwargs=possible_target_kwargs,
                                                    config_backbone_model=config_backbone_model,
                                                    contextual_dataset_names=contextual_dataset_names,
                                                    possible_contextual_kwargs=possible_contextual_kwargs,
                                                    netmob_preprocessing_kwargs=netmob_preprocessing_kwargs
                                                    )


        baselineconfigbuilder = BaselineConfigBuilder(target_data,contextual_dataset_names,dataset_for_coverage,model_name,horizons,freq,REPEAT_TRIAL,SANITY_CHECKER,compilation_modification,add_name_save,)
        dic_configs = baselineconfigbuilder.build_config_single_contextual(dic_configs, possible_target_kwargs, config_backbone_model)

        print(f"Total configurations to test: {len(dic_configs)}")
        for key in dic_configs.keys():
            print(f"  {key}")


        # ==================================================
        # TRAIN ALL CONFIGURATIONS WITH CONTEXTUAL AND BASELINE: 
        loger = loop_train_save_log(loger,dic_configs,init_save_folder = training_save_folder) 


        # ==================================================
        # SAVE RESULTS OF EXPERIMENTS IN A .PY FILE WITH FORMAT SUITABLE FOR ANALYSIS:
        '''
        save the string loger.display_log() in a f"{ROOT}/experiences.pipeline_desag.results.{exp_i}.{exp_i}.py" file to be imported after
        Expected format: 
        results = loger.log_final
        '''
        # if (loger.log_final is not None) and (loger.log_final != ""):
        if not os.path.exists(f"{ROOT}/experiences/pipeline_desag/results/{exp_i}"):
            os.mkdir(f"{ROOT}/experiences/pipeline_desag/results/{exp_i}")
        if not os.path.exists(f"{ROOT}/experiences/pipeline_desag/results/{exp_i}/{exp_i}.py"):
            with open(f"{ROOT}/experiences/pipeline_desag/results/{exp_i}/{exp_i}.py",'w') as f:
                f.write(f'results = {repr(loger.log_final)}')
        else:
            # If file already exists, reload module and append new results to it
            module_path = f"experiences.pipeline_desag.results.{exp_i}.{exp_i}"
            module = importlib.import_module(module_path)
            importlib.reload(module)
            results_saved = module.results
            with open(f"{ROOT}/experiences/pipeline_desag/results/{exp_i}/{exp_i}.py",'w') as f:
                f.write(f'results = {repr(results_saved + loger.log_final)}')

# ==================================================
# BOXPLOT FIGURE :
''' 
Load saved results from f"{ROOT}/experiences.pipeline_desag.results.{exp_i}.{exp_i}.py"
'''

module_path = f"experiences.pipeline_desag.results.{exp_i}.{exp_i}"
module = importlib.import_module(module_path)
importlib.reload(module)
results_saved = module.results
# re._pattern = rf'{model_name}_{target_data}.*?bis'
re._pattern = rf"{model_name}_{target_data}_(?:{'_'.join(contextual_dataset_names)}|calendar).*?bis"
trials = [c[:-4] for c in list(set(re.findall(re._pattern, results_saved)))]
trials = [t for t in trials if not '__e1_h' in t]

if TRIVIAL_TEST:
    trials = [t for t in trials if 'calendar__e' in t and '_h1' in t][:2] + [t for t in trials if '_h1' in t and not 'calendar__e' in t][:1]
    horizons = [1]


print('re._pattern: ',re._pattern)
print('trials found: ',len(trials))
for trial in trials:
    print(f"   {trial}")
# ==================================================
# DESAGREGATED VISUALISATION & SAVE OF FIGURES: 
''' Pour chaque expérience `exp_i` et pour chaque horizon `horizon`: 
    - Récupère la baseline config
    - Récupère l'ensemble des configs dans dic_trials[exp_i]
    - Récupère le model_args associé dans f'{exp_i}/{target_data}_{model_name}'
    - Et pour chacune des config à tester: 
        - Génère le desagregated plot 
        - Print les performances
        - Sauvegarde les figures dans f"{folder_path}/plot/{exp_i}{rainy}"
'''

if True:
    # -- ON RAINY & NON RAINY : 
    dic_bd_metrics_all = get_desagregated_gains( 
        dic_exp_to_names={ 
                exp_i:f'{target_data}_{model_name}'
                },
        dic_trials = {exp_i:trials},
        horizons=horizons,
        comparison_on_rainy_events=comparison_on_rainy_events,
        range_k=range(1,REPEAT_TRIAL+1),
        station_clustering=station_clustering,
        folder_path=f'{current_file_path}/results',
        save_bool=True,
        heatmap= True,
        daily_profile=True,
        dendrogram=True,
        dataset_names =contextual_dataset_names,
        bool_plot = False,
        list_top_k_percent = list_top_k_percent,
        )

    list_rainy = [True, False] if comparison_on_rainy_events else [False]
    trial_id0 = list(dic_bd_metrics_all[exp_i][horizons[0]].keys())[0]
    key_topk0 = list(dic_bd_metrics_all[exp_i][horizons[0]][trial_id0].keys())[0]
    key_rainy0 = list(dic_bd_metrics_all[exp_i][horizons[0]][trial_id0][key_topk0].keys())[0]
    local_temporal_aggs = list(dic_bd_metrics_all[exp_i][horizons[0]][trial_id0][key_topk0][key_rainy0].keys())
    for topk_percent in list_top_k_percent:
        key_topk = int(topk_percent*100) if topk_percent is not None else 100
        for bool_rainy in list_rainy:
            for temporal_agg_i in local_temporal_aggs:
                key_rainy = 'rainy' if bool_rainy else 'all'


                # ---- Get save path :
                path_within_sys = "experiences/pipeline_desag/results"
                base_path = f"{ROOT}/{path_within_sys}"
                if not os.path.exists(f"{base_path}/{exp_i}"):
                    print(f"Creating folder for experiment {exp_i} at path: {base_path}/{exp_i}")
                    os.mkdir(f"{base_path}/{exp_i}")
                if not os.path.exists(f"{base_path}/{exp_i}/{key_rainy}"):
                    print(f"Creating folder for {'rainy events' if bool_rainy else 'all events'} at path: {base_path}/{exp_i}/{key_rainy}")
                    os.mkdir(f"{base_path}/{exp_i}/{key_rainy}")
                if not os.path.exists(f"{base_path}/{exp_i}/{key_rainy}/topk{key_topk}"):
                    print(f"Creating folder for top-{key_topk} at path: {base_path}/{exp_i}/{key_rainy}/topk{key_topk}")
                    os.mkdir(f"{base_path}/{exp_i}/{key_rainy}/topk{key_topk}")
                exp_results_name = f"{exp_i}/{key_rainy}/topk{key_topk}/{temporal_agg_i}"

                saved_results_path= f"{base_path}/{exp_results_name}"
                # ---

                # Load already saved results if exist: 
                py_path = f"{saved_results_path}.py"
                if os.path.exists(py_path):
                    saved_results_path_within_sys = f"{path_within_sys}/{exp_results_name}".replace('/','.')
                    module = importlib.import_module(saved_results_path_within_sys)
                    importlib.reload(module)
                    results_saved = module.results

                else:
                    results_saved = ''
                # -----
                # ---- 
                for trial_id in trials:
                    for h in horizons:
                        dic_metrics = dic_bd_metrics_all[exp_i][h][trial_id+'_bis'][key_topk][key_rainy][temporal_agg_i]
                        rmse_i, mae_i, mase_i, mape_i = dic_metrics['RMSE'], dic_metrics['MAE'], dic_metrics['MASE'], dic_metrics['MAPE']
                        for k_bis in range(1,len(rmse_i)+1):
                            if not f"{trial_id}_bis{k_bis}" in results_saved:
                                str_to_add = f"{trial_id}_bis{k_bis}:   All Steps RMSE = {'{:.3f}'.format(rmse_i[k_bis-1])}, MAE = {'{:.3f}'.format(mae_i[k_bis-1])}, MASE = {'{:.3f}'.format(mase_i[k_bis-1])}, MAPE = {'{:.3f}'.format(mape_i[k_bis-1])}\n"
                                results_saved += str_to_add
                    # ----

                # --- Save results in the .py with format """results = <results_saved>""""
                with open(f"{saved_results_path}.py",'w') as f:
                    f.write(f'results = {repr(results_saved)}')

if True:
    exporter = MetricExporter(results_saved, contextual_dataset_names)
    exporter.export_all(folder_path=save_path_figures, exp_i=exp_i)



    plotting_boxplot_of_trials(trials,
                            exp_i,
                            metrics,
                            folder_path,
                            target_data= target_data,
                            model_name= model_name,
                            dataset_names = contextual_dataset_names,
                            save_path = save_path_figures,
                            n_bis_range = range(1,REPEAT_TRIAL+1)
                            )
