import sys
import os

current_path = os.path.dirname(__file__)
working_dir = os.path.abspath(os.path.join(current_path, '..'))
if working_dir not in sys.path:
    sys.path.insert(0, working_dir)

import pandas as pd
import pickle
from argparse import Namespace

from utils.utilities_DL import get_loss,load_model_and_optimizer,match_period_coverage_with_netmob
from build_inputs.load_subway_in import load_subway_in
from calendar_class import get_time_slots_labels
from constants.paths import file_name


# Data folder:
folder = 'save/HyperparameterTuning'

# Select a id from HP-tuning: 
trial_id = 'subway_in_STGCN_MSELoss_2024_08_21_14_50_2810'

# Load HP-tuning results :
df_hp_tuning =pd.read_csv(f'{working_dir}/{folder}/{trial_id}.csv').head()
model_args = pickle.load(open(f'{working_dir}/{folder}/model_args.pkl','rb'))

# Load common args through all tuning trials:
args = model_args['model'][trial_id]['args']

# Get best config :
metric = '_metric/Loss_model'
best_model = df_hp_tuning.sort_values(metric).iloc[0]

# Set tuned parameter from best config to 'args':
HP_args = [indx.replace('config/', '') for indx in best_model.index if 'config/' in indx]
for arg in HP_args:
    args[arg] = best_model[f'config/{arg}']

# Transform 'dict' to 'Namespace' object: 
args = Namespace(**args)

# Load covergae : 
coverage = match_period_coverage_with_netmob(file_name)

if __name__ == '__main__':
    # Load model with the best config:
    dataset,_,_ = load_subway_in(file_name,args,coverage)
    _,dic_class2rpz,_,_ = get_time_slots_labels(dataset,nb_class = [0,1,2,3])
    loss_function = get_loss(args)
    model,optimizer,scheduler = load_model_and_optimizer(args,dic_class2rpz)

    print(model)