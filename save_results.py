import pandas as pd
import warnings

# cPickle préferable à pickle, mais pas partout disponible 
import os 
import torch 

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import random 
from datetime import date


def get_trial_id(args,fold = None):
    if args.loss_function_type == 'quantile':
        l = 0
    elif args.loss_function_type == 'MSE':
        l = 1
    else:
        raise NotImplementedError(f"{args.loss_function_type} not implemented")

    t = 1 if args.time_embedding is True else 0
    s = 1 if args.scheduler is True else 0

    if fold is None:
        trial_id1 = f"{random.randint(1,100000)}_{args.model_name}_F{args.K_fold}f"
        trial_id2 = f"_{args.batch_size}_{args.epochs}_{l}{t}{s}_{date.today().strftime('%d_%m_%Y')}"
        return(trial_id1,trial_id2)
    
    else : 
        trial_id = f"{random.randint(1,100000)}_{args.model_name}_F{args.K_fold}f{fold}_{args.batch_size}_{args.epoch}_{l}{t}{s}_{date.today().strftime('%d_%m_%Y')}"
        return(trial_id)

def load_json_file(save_dir):
    ''' Load Json-file containing ID of DeepLearning trial and all the usefull arguments'''
    # if json_file doesn't exist, build it 
    json_save_path = f"{save_dir}model_args.pkl"
    if not os.path.exists(json_save_path):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        json_file = {'model':{}}
        pickle.dump(json_file,open(json_save_path,'wb'))
    else:
        # Load json_file
        json_file = pickle.load(open(json_save_path,'rb'))

    return(json_file)

def update_json(args,json_file,trial_id,performance,save_dir):
    '''Add the trial and its metrics to the save file'''
    # Fill json_file
    dic_args = vars(args)

    # update 
    json_file['model'][trial_id] = {'args': dic_args,'performance': performance}

    # Save Json_file
    pickle.dump(json_file,open(f"{save_dir}model_args.pkl",'wb'))


def save_best_model_and_update_json(checkpoint,trial_id,performance,args,save_dir = 'save/best_models/'):
    ''' '''
    json_file = load_json_file(save_dir)
    update_json(args,json_file,trial_id,performance,save_dir)
    torch.save(checkpoint, f"{save_dir}{trial_id}_best_model.pkl")
    



def time_embedding2dict(args):
    dict1 = dict(CalendarClass = args.calendar_class, 
        Position = args.position, 
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
    
def Dataset_get_save_folder(args,K_fold = None, fold = None):
    # Define the name of the save folder 
    W_D_H_S = f"W{args.W}_D{args.D}_H{args.H}_S{args.step_ahead}"
    prop_station = f"train{str(args.train_prop).replace('.','')}_val{str(args.valid_prop).replace('.','')}_cal{str(args.calib_prop).replace('.','')}_single_station{args.single_station}"

    if K_fold is None:
        K_fold = args.K_fold
    
    if fold is None:
        fold = 0
    B_K_validation = f"B{args.batch_size}_K_fold{K_fold}_Cclass{args.calendar_class}_validation{args.validation}_fold{fold}"

    save_folder = f"data/loading/{W_D_H_S}/{prop_station}/{B_K_validation}/"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder,exist_ok = True)

    return(save_folder)

