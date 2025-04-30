# GET PARAMETERS
import os 
import sys
import torch 
# Get Parent folder : 

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from examples.benchmark import local_get_args
from examples.train_and_visu_non_recurrent import evaluate_config,train_the_config,get_ds
from high_level_DL_method import load_optimizer_and_scheduler
from dl_models.full_model import full_model
from trainer import Trainer
from constants.config_by_datasets import dic_config

# Init:
#['subway_indiv','tramway_indiv','bus_indiv','velov','criter']
target_data = 'subway_in' #'subway_in'  # PeMS03 # PeMS04 # PeMS07 # PeMS08 # METR_LA 
dataset_names = ['subway_in','subway_out'] #['PeMS03'] #['subway_in'] ['subway_in','subway_indiv'] #["subway_in","subway_out"] # ['subway_in','netmob_POIs_per_station'],["subway_in","subway_out"],["subway_in","calendar"] # ["subway_in"] # ['data_bidon'] # ['METR_LA'] # ['PEMS_BAY']
dataset_for_coverage = ['subway_in','netmob_image_per_station']#['subway_in','subway_indiv'] # ['subway_in','netmob_image_per_station'] #  ['data_bidon','netmob'] #  ['subway_in','netmob']  # ['METR_LA'] # ['PEMS_BAY']
model_name = 'STGCN' # 'STGCN', 'ASTGCN' # 'STGformer'
#station = ['BEL','PAR','AMP','SAN','FLA']# ['BEL','PAR','AMP','SAN','FLA']   # 'BON'  #'GER'
# ...

# Modif 
modification = {'target_data': target_data, 
                'freq': '15min', #'15min', # '5min'
                'use_target_as_context': False,
                'data_augmentation': False,
                'step_ahead':4,
        
                'epochs' : 100, #100

                # Contextual data:


                'lr': 0.00105, # 5e-5,# 4e-4,
                'weight_decay': 0.0188896655584368, # 0.05,
                'dropout': 0.271795372610271, # 0.15,
                'Kt': 2,
                'stblock_num': 3,
                'gso_type': 'sym_renorm_adj',
                'temporal_h_dim': 256,
                'spatial_h_dim': 32,
                'output_h_dim': 16,
                'scheduler': True,  # None
                'torch_scheduler_milestone': 28.0, #5,
                'torch_scheduler_gamma': 0.9958348861339396, # 0.997,
                'torch_scheduler_lr_start_factor': 0.8809942312067847, # 1,

                #'set_spatial_units':  station,   
                'adj_type':'corr',
                'threshold': 0.7,
                'learnable_adj_matrix' : False, # True      

                'stacked_contextual': True, # True # False
                'temporal_graph_transformer_encoder': False,
                'compute_node_attr_with_attn' : False, # True ??

                           }

if target_data in dic_config.keys():
    modification.update(dic_config[target_data])


args_init = local_get_args(model_name,
                    args_init = None,
                    dataset_names=dataset_names,
                    dataset_for_coverage=dataset_for_coverage,
                    modification = modification)


def main():
    ds,args,trial_id,save_folder,df_loss = get_ds(modification=modification,args_init=args_init)
    model = full_model(ds, args).to(args.device)
    optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
    trainer = Trainer(ds,model,args,optimizer,loss_function,scheduler = scheduler,show_figure = False,trial_id = trial_id, fold=0,save_folder = save_folder)
    trainer.train_and_valid(normalizer = ds.normalizer, mod = 1000,mod_plot = None) 
    return trainer,ds,model,args


if __name__ == "__main__":
    # Run the script
    trainer,ds,model,args = main()
    print("Training completed successfully.")
    print(trainer.performance)