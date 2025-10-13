import os 
import sys
import torch 
import importlib

import torch._dynamo as dynamo; dynamo.graph_break()
torch._dynamo.config.verbose=True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..','..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from pipeline.utils.loger import LOG
from experiences.common_parameter import possible_target_kwargs, weather_possible_contextual_kwargs, model_configurations,feature_extractor_model_configurations,subway_possible_contextual_kwargs
from experiences.common_parameter import REPEAT_TRIAL,percent_train_size,modif_percent_train_size
from experiences.compilation_parameter import compilation_modification
from experiences.loop_train_save_log import loop_train_save_log
from examples.benchmark import local_get_args
from examples.train_and_visu_non_recurrent import evaluate_config,train_the_config,get_ds
from pipeline.high_level_DL_method import load_optimizer_and_scheduler
from pipeline.dl_models.full_model import full_model
from pipeline.trainer import Trainer
from pipeline.utils.loger import LOG
from pipeline.utils.rng import set_seed
from examples.train_model import main 
from examples.train_model_on_k_fold_validation import save_model_metrics,get_conditions,keep_track_on_metrics,init_metrics
from constants.paths import SAVE_DIRECTORY, FOLDER_PATH


# --- Init ---  (set horizon, freq, ...)
# Set seed : NO 

dic_configs = {}


init_save_folder = 'K_fold_validation/training_wo_HP_tuning/Exp5'
device = torch.device('cuda:0')
# REPEAT_TRIAL  = 1 
freq = '15min' #'15min'  
horizons = [4,1] # [4]  #[1,4]



model_name = 'STAEformer'
config_backbone_model = model_configurations[model_name]
config_backbone_model['epochs'] = 50 #1  #50 #80
compilation_modification['torch_compile'] = 'compile' # 'compile'  # False 
compilation_modification['device'] = device
weather_contextual_kwargs = weather_possible_contextual_kwargs['early_fusion']['repeat_t_proj']


# --- Set Combinaison (Target, Conrtextual Datasets)
for target_data in ['bike_out','subway_out']:
    for horizon in horizons:
        loger = LOG()
        if target_data == 'bike_out':
            L_contextual_dataset_names = [[],['weather']]
            dataset_for_coverage = ['bike_out']
        if target_data == 'subway_out':
            L_contextual_dataset_names = [[],['subway_in'],['subway_in','weather']]
            dataset_for_coverage = ['subway_in']

        for contextual_dataset_names in L_contextual_dataset_names:

            # --- Get Associated contextual kwargs (only one single here)
            contextual_kwargs_i = subway_possible_contextual_kwargs['early_fusion']['shared_embedding']
            if 'weather' in contextual_dataset_names:
                if len(contextual_dataset_names)>1:
                    fusion_type = 'early_fusion'
                    feature_extractor_type = 'shared_embedding_repeat_t_proj'
                else:
                    fusion_type = 'early_fusion'
                    feature_extractor_type = 'repeat_t_proj'
            else: 
                fusion_type = 'early_fusion'
                feature_extractor_type = 'shared_embedding'
            # contextual_kwargs_i = {'early_fusion': {'shared_embedding': subway_possible_contextual_kwargs['early_fusion']['shared_embedding']}}


            # --- Update contextual_kwargs according to the selected contextual datasets
            contextual_kwargs ={'subway_out':contextual_kwargs_i,
                                'subway_in':contextual_kwargs_i,
                                'subway_in_subway_out':contextual_kwargs_i,
                                'weather':weather_contextual_kwargs}
            
            if 'weather' not in contextual_dataset_names:
                contextual_kwargs.pop('weather',None)  
            if 'subway_in' not in contextual_dataset_names:
                contextual_kwargs.pop('subway_in',None)  
            if 'subway_out' not in contextual_dataset_names:
                contextual_kwargs.pop('subway_out',None)
            if 'subway_in_subway_out' not in contextual_dataset_names:
                contextual_kwargs.pop('subway_in_subway_out',None)  
            # --- 

            for percent,modif_percent in percent_train_size.items():
                modif_percent = modif_percent_train_size(target_data,freq,percent,modif_percent,dataset_for_coverage)

                for n_bis in range(1,REPEAT_TRIAL+1): # range(1,6):
                    dataset_names =  [target_data] +contextual_dataset_names+ ['calendar']
                    if len(contextual_dataset_names)>0:
                        name_i = f"{model_name}_{'_'.join(dataset_names)}_{fusion_type}_{feature_extractor_type}_{percent}"
                    else:
                        name_i = f"{model_name}_{target_data}_{percent}"
                    name_i_end = f"_e{config_backbone_model['epochs']}_h{horizon}_bis{n_bis}"
                    name_i = f"{name_i}_{name_i_end}"

                    config_i =  {'target_data': target_data,
                                'dataset_names': dataset_names,
                                'model_name': model_name,
                                'dataset_for_coverage': [target_data],
                                'freq': freq,
                                'horizon_step': horizon,
                                'step_ahead': horizon,
                                'target_kwargs' : {target_data: possible_target_kwargs[target_data]},
                                'contextual_kwargs' : contextual_kwargs,
                                'denoising_names':[],
                                } 
                    config_i.update(config_backbone_model)
                    config_i.update(compilation_modification)
                    config_i.update(modif_percent)

                    dic_configs[name_i] = config_i
                    
        loop_train_save_log(loger,dic_configs,init_save_folder = init_save_folder) 