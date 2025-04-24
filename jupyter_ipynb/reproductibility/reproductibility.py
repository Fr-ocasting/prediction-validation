import os
import sys
import torch
import numpy as np
import random
import pandas as pd  
import warnings
warnings.simplefilter("error", category=RuntimeWarning)

# === FIXE LA SEED POUR REPRODUCTIBILITÉ ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === CHEMIN ET IMPORTS ===
current_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_path, '..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from examples.train_and_visu_non_recurrent import evaluate_config,train_the_config,get_ds
from examples.benchmark import local_get_args
from high_level_DL_method import load_optimizer_and_scheduler
from dl_models.full_model import full_model
from trainer import Trainer

# === PARAMÈTRES GÉNÉRAUX ===
SEED = 42
EPOCHS = 1  # une seule epoch


def get_modification(dataset_names):
    # Définir la base de la modification
    modification = {
        'epochs': EPOCHS,
        'lr': 5e-5,
        'weight_decay': 0.05,
        'dropout': 0.15,
        'scheduler': None,
        'adj_type': 'corr',
        'threshold': 0.7,
        'stacked_contextual': True,
        'target_data': 'subway_in',
        'compute_node_attr_with_attn': False,
        'use_target_as_context': False,
    }

    if "netmob_POIs" in dataset_names:
        modification.update({
            'NetMob_only_epsilon': True,
            'NetMob_selected_apps': ['Google_Maps'],
            'NetMob_transfer_mode': ['DL'],
            'NetMob_selected_tags': ['station_epsilon100'],
            'NetMob_expanded': ''
        })
    return modification

def load_inputs(model_name,dataset_names,dataset_for_coverage):
    # Init args
    modification = get_modification(dataset_names=dataset_names)
    args_init = local_get_args(model_name,args_init=None,dataset_names=dataset_names,dataset_for_coverage=dataset_for_coverage,modification=modification)

    # Load ds
    ds,args,trial_id,save_folder,df_loss = get_ds(modification=modification,args_init=args_init)
    return ds,args,trial_id,save_folder

# === FONCTION POUR UNE CONFIG SPÉCIFIQUE ===
def run_test(model_names, dataset_names,dataset_for_coverage):
    set_seed(SEED)
    df = pd.DataFrame()
    ds,args,trial_id,save_folder = load_inputs(model_names[0],dataset_names,dataset_for_coverage)
    for model_name in model_names:
        print(f"\n=== TESTING {model_name} on {dataset_names} ===")
        args.model_name = model_name
        model = full_model(ds, args).to(args.device)
        optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
        trainer = Trainer(ds,model,args,optimizer,loss_function,scheduler = scheduler,show_figure = False,trial_id = trial_id, fold=0,save_folder = save_folder)
        trainer.train_and_valid(normalizer = ds.normalizer, mod = 1000,mod_plot = None) 

        mse_test = trainer.performance['test_metrics']['mse']
        mse_valid = trainer.performance['valid_metrics']['mse']
        df = pd.concat([df,pd.DataFrame({'mse_test': [mse_test], 'mse_valid': [mse_valid]}, index=[model_name])], axis=0)

    print("=== TEST COMPLETED ===")
    return ds,trainer,df

# === LANCEMENT GLOBAL ===
if __name__ == "__main__":
    model_names = ['STGCN','ASTGCN']   # 'DCGRU','MTGNN','LSTM','RNN','CNN','GRU'
    dataset_for_coverage = ['subway_in', 'netmob_image_per_station']
    dataset_names= ["subway_in","subway_out"] #     ["subway_in", "netmob_POIs"] #    ["subway_in"]

    ds,last_trainer,df = run_test(model_names, dataset_names,dataset_for_coverage)
    
    # Afficher le DataFrame des résultats
    print("\n=== RÉSULTATS ===")
    print(df)
    
    # Créer un DataFrame de référence pour vérification
    checking = pd.DataFrame({
        'mse_test': [5988.421875, 48209.429688],
        'mse_valid': [6884.651855, 71904.921875]
    }, index=['STGCN', 'ASTGCN'])

    print("\n=== HAS TO BE EQUAL TO: ===")
    print(checking)

    print("\n=== ABSOLUTE DIFFERENCE BETWEEN BOTH DF: ===")
    print(abs(checking-df))