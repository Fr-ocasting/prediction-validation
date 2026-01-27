# ==================================================
# IMPORT
import re 
import os 
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
from experiences.common_parameter import possible_target_kwargs, weather_possible_contextual_kwargs, model_configurations,feature_extractor_model_configurations
from experiences.common_parameter import REPEAT_TRIAL,netmob_preprocessing_kwargs
from experiences.get_desagregated_plot import get_desagregated_gains
from experiences.compilation_parameter import compilation_modification
from experiences.loop_train_save_log import loop_train_save_log
from experiences.pipeline_desag.build_config_single_contextual import ConfigBuilder
from experiences.pipeline_desag.build_baseline_config import BaselineConfigBuilder
from experiences.pipeline_desag.utils import plotting_boxplot_of_trials
from constants.paths import ROOT
inside_saved_folder = 'K_fold_validation/training_wo_HP_tuning'
folder_path = f"{ROOT}/save/{inside_saved_folder}"

# ==================================================
# SET PARAMETERS OF THE EXPERIMENT AND LOGER: 
'''
- Do not set seed to have different initialization for each trial
- SANITY_CHECKER: If True, Keep track of the gradients and the weights during the training to detect possible problems.
'''
exp_i = 'pipeline_subway_in_X'

training_save_folder = f'{inside_saved_folder}/{exp_i}' # f'K_fold_validation/training_wo_HP_tuning/{exp_i}' 
save_path_figures = f'{current_file_path}/results/plot/{exp_i}'
device = torch.device('cuda:0')
add_name_save = '' #'_clipping'  # ''  # '_trial2'

freq = '15min' #'15min'  
horizons = [1] # [4]  #[1,4]
model_name = 'STAEformer'
target_data = 'subway_in' 
station_clustering = True
contextual_dataset_names = ['subway_out'] # ['netmob_POIs'] #['subway_out']
assert len(contextual_dataset_names) == 1, "Only one contextual dataset at a time is allowed for this pipeline. Otherwise, update 'build_config_single_contextual.py' accordingly. "
weather_contextual_kwargs = weather_possible_contextual_kwargs['early_fusion']['repeat_t_proj']

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
TRIVIAL_TEST = True
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


    
# ==================================================
# LOAD CONFIGURATIONS TO TEST:
if (target_data == 'subway_in') or (target_data == 'subway_out'):
    if contextual_dataset_names == ['netmob_POIs']:
        from experiences.pipeline_desag.subway_in_pred.netmob_POIs_contextual import get_possible_contextual_kwarg
        possible_contextual_kwargs = get_possible_contextual_kwarg(add_name_save)

    elif contextual_dataset_names == ['subway_out']:
        from experiences.pipeline_desag.subway_in_pred.subway_out_contextual import get_possible_contextual_kwarg
        possible_contextual_kwargs = get_possible_contextual_kwarg(add_name_save)
    else:
        raise NotImplementedError(f'Contextual dataset {contextual_dataset_names} not implemented for target_data {target_data} ')
else:
    raise NotImplementedError(f'Target data {target_data} not implemented yet. ')

if TRIVIAL_TEST:
    k0 = list(possible_contextual_kwargs.keys())[0]
    k1 = list(possible_contextual_kwargs[k0].keys())[0]
    v1 = possible_contextual_kwargs[k0][k1]
    possible_contextual_kwargs = {k0 : {k1:v1}}

print('\n------------------------------- CONFIGURATIONS ----------------------------------\n')
for fusion_type in possible_contextual_kwargs.keys():
    print(f'--- Fusion type: {fusion_type} ---')
    for trial_id in possible_contextual_kwargs[fusion_type].keys():
        print(f'    {trial_id}')
print('\n---------------------------------------------------------------------------------\n')
# ==================================================
# LOAD CONFIG DICTIONARY: 
configbuilder = ConfigBuilder(target_data,model_name,horizons,freq,REPEAT_TRIAL,SANITY_CHECKER,compilation_modification)
dic_configs = configbuilder.build_config_single_contextual(
                                             dic_configs = {},
                                             possible_target_kwargs=possible_target_kwargs,
                                             config_backbone_model=config_backbone_model,
                                             contextual_dataset_names=contextual_dataset_names,
                                             possible_contextual_kwargs=possible_contextual_kwargs,
                                             weather_contextual_kwargs=weather_contextual_kwargs,
                                             netmob_preprocessing_kwargs=netmob_preprocessing_kwargs
                                             )


baselineconfigbuilder = BaselineConfigBuilder(target_data,model_name,horizons,freq,REPEAT_TRIAL,SANITY_CHECKER,compilation_modification,add_name_save)
dic_configs = baselineconfigbuilder.build_config_single_contextual(dic_configs,possible_target_kwargs,
                                                                   config_backbone_model)


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
re._pattern = rf'{model_name}.*?bis'
trials = [c[:-4] for c in list(set(re.findall(re._pattern, results_saved)))]

plotting_boxplot_of_trials(trials,
                           exp_i,
                           metrics,
                           folder_path,
                           dic_exp_to_names={exp_i:f'{target_data}_{model_name}'} ,# {exp_i:exp_i},
                           save_path = save_path_figures,
                           n_bis_range = range(1,REPEAT_TRIAL+1)
                           )
# dic_df_horizons_init,dic_df_horizons =  local_plot_boxplot_metrics(experiences,metrics,folder_path,dic_exp_to_names,palette,legend_groups,configs_to_keep=configs_to_keep,fusion_type_to_keep =fusion_type_to_keep)

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

# -- ON NON RAINY
get_desagregated_gains(dic_exp_to_names={exp_i:exp_i},
                       dic_trials = {exp_i:trials},
                       horizons=horizons,
                       comparison_on_rainy_events=False,
                       range_k=range(1,REPEAT_TRIAL+1),
                       station_clustering=station_clustering,
                       save_folder_path=save_path_figures,
                       save_bool=True,
                       heatmap= True,
                       daily_profile=True,
                       dendrogram=True,
                       )

# -- ON RAINY 
get_desagregated_gains(dic_exp_to_names={exp_i:exp_i},
                       dic_trials = {exp_i:trials},
                       horizons=horizons,
                       comparison_on_rainy_events=True,
                       range_k=range(1,REPEAT_TRIAL+1),
                       station_clustering=station_clustering,
                       save_folder_path=save_path_figures,
                       save_bool=True,
                       heatmap= True,
                       daily_profile=True,
                       dendrogram=True,
                       )

# ==================================================
# SAVE LATEX TABLES OF RESULTS WITH POURCENTAGE: 

# ==================================================
# S ASSURER QUE LES SAUVEGARDES DE MODELES NE SONT JAMAIS ECRASEES MAIS BIEN SAUVEGARDEES DANS DES DOSSIERS DIFFERENTS