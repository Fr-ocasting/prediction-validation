import pandas as pd
import warnings

# cPickle préferable à pickle, mais pas partout disponible 
import os 
import torch 
import sys
import os

# Get Parent folder : 
current_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import random 
from datetime import datetime
#from datetime import date

def get_date_id():
    return(f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}_{random.randint(1,100000)}")


def get_trial_id(args):
    date_id = get_date_id()
    dataset_names = '_'.join(args.dataset_names)

    models_names = [args.model_name] if args.model_name is not None else []
    for dataset in args.dataset_names:
        if dataset in args.contextual_kwargs.keys() and 'args_vision' in args.contextual_kwargs[dataset].keys() and len(vars(args.contextual_kwargs[dataset]['args_vision']))>0: 
            models_names.append(args.contextual_kwargs[dataset]['args_vision'].model_name)
    if len(vars(args.args_embedding))>0:
        models_names.append('TE') 

    model_names = '_'.join(models_names)

    trial_id =  f"{dataset_names}_{model_names}_{args.loss_function_type}Loss_{date_id}_F{args.K_fold}"
    return trial_id

def load_json_file(path_folder):
    json_save_path = f'{path_folder}/model_args.pkl'
    ''' Load Json-file containing ID of DeepLearning trial and all the usefull arguments'''
    # if json_file doesn't exist, build it 
    if not os.path.exists(json_save_path):
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
        json_file = {'model':{}}
        pickle.dump(json_file,open(json_save_path,'wb'))
    else:
        # Load json_file
        #json_file = pickle.load(open(json_save_path,'rb'))
        with open(json_save_path, 'rb') as f:
            json_file = pickle.load(f)

    return(json_file)

def update_json(args,json_file,trial_id,performance,json_save_path):
    '''Add the trial and its metrics to the save file'''
    # Fill json_file
    dic_args = vars(args)

    # update 
    json_file['model'][trial_id] = {'args': dic_args,'performance': performance}

    # Save Json_file (write binary)
    with open(json_save_path, 'wb') as f:
        pickle.dump(json_file, f)


def save_best_model_and_update_json(checkpoint,trial_id,performance,args,save_dir,update_checkpoint=True):
    ''' '''
    path_folder = f"{parent_dir}/{save_dir}"
    json_file = load_json_file(path_folder)
    update_json(args,json_file,trial_id,performance,json_save_path = f"{path_folder}/model_args.pkl")
    if update_checkpoint: 
        torch.save(checkpoint, f"{path_folder}/{trial_id}.pkl")
    

    



def time_embedding2dict(args):
    dict1 = dict(CalendarClass = args.calendar_class, 
        Specific_lr = args.specific_lr,
        Type_calendar = args.type_calendar,
        Multi_Emb = args.multi_embedding,
        TE_transfer = args.TE_transfer,
        single_station = args.single_station
        )
    
    return(dict1)

def results2dict(args,epoch,picp,mpiw,valid_loss,train_loss):
    '''Set results in a dictionnary. Will be used as a row for results dataframe'''
    dict_row = {}
    dict1 = time_embedding2dict(args)
    dict2 = dict(epoch = epoch, PICP = picp, MPIW = mpiw, valid_loss = valid_loss, train_loss = train_loss)
    for dic in [dict1,dict2]:
        dict_row.update(dic)
    return(dict_row)

def Multi_results2dict(args,mean_picp,mean_mpiw,dict3,dict4):
    '''Same then resuls2dict, but change the name of PICP and MPIW as it's for several folds '''
    dict_row = {}
    dict1 = time_embedding2dict(args)
    dict2 = dict(PICP_mean = mean_picp.item(),MPIW_mean = mean_mpiw.item())
    for dic in [dict1,dict2,dict3,dict4]:
        dict_row.update(dic)
    return(dict_row)


def update_results_df(results_df, dict_row):
    df_row = pd.DataFrame.from_records([dict_row])
    if (not results_df.empty) &(not df_row.empty):
        with warnings.catch_warnings():
            # TODO: pandas 2.1.0 has a FutureWarning for concatenating DataFrames with Null entries
            warnings.filterwarnings("ignore", category=FutureWarning)
            return (pd.concat([results_df,df_row]))
    elif (not results_df.empty) & (df_row.empty):
        return results_df
    elif (results_df.empty) & (not df_row.empty):
        return df_row
    else:
        return results_df



def build_results_df(results_df,args, mean_picp,mean_mpiw,dict2,dict3):
    dict_row = Multi_results2dict(args,mean_picp,mean_mpiw,dict2,dict3)
    results_df = update_results_df(results_df, dict_row)
    return(results_df)


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def read_object(filename):
    with open(filename, 'rb') as f:  # Overwrites any existing file.
        return(pickle.load(f))
    
def Dataset_get_save_folder(args,K_fold = None, fold = None, netmob = False):
    # Define the name of the save folder 
    W_D_H_S = f"W{args.W}_D{args.D}_H{args.H}_S{args.step_ahead}"
    prop_station = f"train{str(args.train_prop).replace('.','')}_val{str(args.valid_prop).replace('.','')}_cal{str(args.calib_prop).replace('.','')}_single_station{args.single_station}"

    if K_fold is None:
        K_fold = args.K_fold
    
    if fold is None:
        fold = 0
    B_K_validation = f"B{args.batch_size}_K_fold{K_fold}_Cclass{args.calendar_class}_validation{args.validation}_fold{fold}"

    save_folder = f"data/loading/{W_D_H_S}/{prop_station}/{B_K_validation}/"

    if netmob:
        save_folder = f"{save_folder}netmob/"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder,exist_ok = True)

    return(save_folder)

