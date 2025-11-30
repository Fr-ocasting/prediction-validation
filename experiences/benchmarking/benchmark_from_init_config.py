
# GET PARAMETERS
import os 
import sys
import torch 
import importlib
import torch._dynamo as dynamo; dynamo.graph_break()
torch._dynamo.config.verbose=True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True
# Get Parent folder : 

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from pipeline.utils.loger import LOG
from experiences.common_parameter import possible_target_kwargs, weather_possible_contextual_kwargs, model_configurations
from experiences.common_parameter import REPEAT_TRIAL
from experiences.compilation_parameter import compilation_modification
from experiences.loop_train_save_log import loop_train_save_log

loger = LOG()

# --- Init ---  (set horizon, freq, ...)
# Set seed : NO 

init_save_folder = 'K_fold_validation/training_wo_HP_tuning/BenchmarkInitConfig'
device = torch.device('cuda:1')

freq = '15min'  
horizons = [4,1]  #[1,4]
target_data = 'subway_in' 
BATCH_SIZE = 128
EPOCHS = 100

dataset_for_coverage = target_data
# REPEAT_TRIAL = 1 


dic_configs = {}
for dataset_names in [[target_data] + ['calendar'],[target_data]]:
    for horizon in horizons:
        for model_name in ['GMAN']: # ,
        # Fonctionnels :  'DCRNN','RNN','LSTM','GRU','STGCN','STAEformer'
        # A faire: 'MTGNN', 'ASTGCN','CNN'
            
            if 'calendar' == dataset_names[-1] and not(model_name in ['STAEformer','GMAN']):
                updated_dataset_names = dataset_names.copy()
                updated_dataset_names[-1] = 'calendar_embedding'
            else:
                updated_dataset_names = dataset_names.copy()

            # Load Config: 
            path = os.path.join(parent_dir, f"constants/config_by_datasets/{target_data}/{model_name}/{'_'.join(updated_dataset_names)}.py")
            abs_path = path.replace('/', os.sep)
            spec = importlib.util.spec_from_file_location("config_module", path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            config_backbone_model = config_module.config
            # ---


            for n_bis in range(1,REPEAT_TRIAL+1): # range(1,6):
                
                name_i = f"{model_name}_{'_'.join(updated_dataset_names)}"
                name_i_end = f"_e{config_backbone_model['epochs']}_h{horizon}_bis{n_bis}"
                name_i = f"{name_i}_{name_i_end}"

                config_i =  {'target_data': target_data,
                            'dataset_names': updated_dataset_names,
                            'model_name': model_name,
                            'dataset_for_coverage': [target_data],
                            'freq': freq,
                            'horizon_step': horizon,
                            'step_ahead': horizon,
                            'target_kwargs' : {target_data: possible_target_kwargs[target_data]},
                            'contextual_kwargs' : {},
                            'denoising_names':[],
                            } 
                config_i.update(compilation_modification)
                config_i.update(config_backbone_model)
                


                config_i['device'] = device

                config_i['torch_compile'] = False 
                config_i['batch_size'] = BATCH_SIZE
                config_i['epochs'] = EPOCHS
                # config_i['epochs'] = 1

                dic_configs[name_i] = config_i

loop_train_save_log(loger,dic_configs,init_save_folder = init_save_folder) 