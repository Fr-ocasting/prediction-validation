import pandas as pd
import warnings

# cPickle préferable à pickle, mais pas partout disponible 
import os 

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle



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

