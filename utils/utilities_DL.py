import numpy as np
import torch 
import torch.nn as nn 
import pandas as pd
from datetime import datetime
import argparse
from torch.optim import SGD,Adam,AdamW
try:
    from torch.optim.lr_scheduler import LinearLR,ExponentialLR,SequentialLR
except:
    print(f'Pytorch version {torch.__version__} does not allow you to use lr-scheduler')

# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal import: 
from constants.config import optimizer_specific_lr, get_config_embed, get_parameters
from DL_class import QuantileLoss
from dataset import  DataSet
from TE_transfer_learning import TE_transfer
from dl_models.full_model import full_model


def get_small_ds(small_ds,coverage,args):
    if small_ds:
        args.time_slot_limit = 1500
        coverage = coverage[: args.time_slot_limit]
        args.W = 0
        args.D = 0
        print(f'Seulement les { args.time_slot_limit} premiers time-slots sont utilisés.')
    else:
        args.time_slot_limit = None
    return(coverage,args)


def forward_and_display_info(model,inputs):
    nb_total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model parameters: ',nb_total_param)
    output =model(inputs)
    print('input size: ',inputs.size())

    if type(output) != tuple: 
        print('output size: ',output.size(),'\n')
    else:
        print('output size: ',output[0][0].size(),'\n')
    print(model)
    return(output)


def match_period_coverage_with_netmob(filename):
    if (filename == 'subway_IN_interpol_neg_15_min_2019_2020.csv'):
        coverage_dataset = pd.date_range(start='01/01/2019', end='01/01/2020', freq='15min')[:-1]
    else:
        raise ValueError("The coverage period of this filename has not been defined")

    coverage_netmob =  pd.date_range(start='03/16/2019', end='06/1/2019', freq='15min')[:-1]
    coverage = list(set(coverage_dataset)& set(coverage_netmob))
    
    if len(coverage) != len(coverage_netmob):
        raise ValueError("Coverage period from dataset doesn't match the NetMob coverage period")
    else:
        return(coverage_netmob)


def get_args_embedding(args,nb_words_embedding):
    if args.time_embedding:
        config_Tembed = get_config_embed(nb_words_embedding,embedding_dim = args.embedding_dim,position = args.position)
        args_embedding = get_parameters(config_Tembed,description = 'TimeEmbedding')
    else:
        args_embedding = argparse.ArgumentParser(description='TimeEmbedding').parse_args(args=[])
    return(args_embedding)


def get_model_loss_args_emb_opts(args,nb_words_embedding,dic_class2rpz,n_vertex = None,args_vision = None):
    args.num_nodes = n_vertex
    loss_function = get_loss(args.loss_function_type,args)
    args_embedding = get_args_embedding(args,nb_words_embedding)

    model,opt,sched = load_model_and_optimizer(args,args_embedding,dic_class2rpz,args_vision)

    if args.TE_transfer:
        if os.path.exists(f'{args.abs_path}data/Trained_Time_Embedding{args.embedding_dim}.pkl'):
            model  = TE_transfer(model,n_vertex,args,model_dir = 'data/')
        else:
            print('TE impossible')
    
    return(loss_function,model,opt,sched,args_embedding)
    

def get_MultiModel_loss_args_emb_opts(args,nb_words_embedding,dic_class2rpz,n_vertex = None,arg_vision = None):
    args.num_nodes = n_vertex
    loss_function = get_loss(args.loss_function_type,args)
    args_embedding = get_args_embedding(args,nb_words_embedding)

    model_opt_sched_list = [load_model_and_optimizer(args,args_embedding,dic_class2rpz,args_vision) for _ in range(args.K_fold)]
    Model_list = [model_opt[0] for model_opt in model_opt_sched_list]
    if args.TE_transfer:
        if os.path.exists(f'{args.abs_path}data/Trained_Time_Embedding{args.embedding_dim}.pkl'):
            Model_list = [TE_transfer(model,n_vertex,args,model_dir = 'data/') for model in Model_list]
        else:
            print('TE impossible')
                                      
        
    Optimizer_list = [model_opt[1] for model_opt in model_opt_sched_list]
    Scheduler_list = [model_opt[2] for model_opt in model_opt_sched_list]
    return(loss_function,Model_list,Optimizer_list,Scheduler_list,args_embedding)


def get_associated__df_verif_index(dataset,date,iloc):
    mask = (dataset.df_verif.iloc[:,iloc] == date)
    try:
        associated_index = dataset.df_verif[mask].index.item()
    except:
        associated_index = None
    return(associated_index)


def find_nearest_date(date_series, date, inferior=True):
    """
    Find the nearest index of the timestamp <= or >= 'date' in 'date_series'.
    Parameters:
        date_series (pd.Series): A series of timestamps.
        date (pd.Timestamp): The reference timestamp.
        inferior (bool): If True, search for the nearest date <= 'date', else >= 'date'.
    Returns:
        int or None: The index of the nearest date, or None if not found.
    """
    # Calculating the difference
    diff = date_series - date
    
    if inferior:
        # Filtering to get the nearest <= date
        filtered_series = diff[diff <= pd.Timedelta(0)]
        if not filtered_series.empty:
            nearest_index = filtered_series.idxmax()
        else:
            return None,None
    else:
        # Filtering to get the nearest >= date
        filtered_series = diff[diff >= pd.Timedelta(0)]
        if not filtered_series.empty:
            nearest_index = filtered_series.idxmin()
        else:
            return None,None
        
    nearest_indice = date_series.index.get_loc(nearest_index)
    
    return nearest_index,nearest_indice

def load_model_and_optimizer(args,args_embedding,dic_class2rpz,args_vision=None):
    model =  full_model(args,args_embedding,dic_class2rpz,args_vision).to(args.device)
    #model = load_model(args,args_embedding,dic_class2rpz).to(args.device)
    # Config optimizer:
    optimizer = choose_optimizer(model,args)
    scheduler = load_scheduler(optimizer,args)

    print('number of total parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    print('number of trainable parameters: {}'.format(sum([p.numel() for p in model.parameters() if p.requires_grad])))
    return(model,optimizer,scheduler)

def get_DataSet_and_invalid_dates(abs_path,folder_path,file_name,W,D,H,step_ahead,single_station = False,coverage_period = None):
    df,invalid_dates,time_step_per_hour = load_raw_data(abs_path,folder_path,file_name,single_station = single_station)
    if coverage_period is not None:
        df = df.loc[coverage_period]
        invalid_dates = list(set(invalid_dates) & set(coverage_period))
    dataset = DataSet(df,time_step_per_hour=time_step_per_hour, Weeks = W, Days = D, historical_len= H,step_ahead=step_ahead)
    print(f"coverage period: {df.index.min()} - {df.index.max()}")
    return(dataset,invalid_dates)

def load_raw_data(abs_path,folder_path,file_name,single_station = False,):
    if (file_name == 'preprocessed_subway_15_min.csv') | (file_name == 'subway_IN_interpol_neg_15_min_2019_2020.csv') | (file_name=='subway_IN_interpol_neg_15_min_16Mar2019_1Jun2020.csv'):
        subway_in = pd.read_csv(abs_path + folder_path+file_name,index_col = 0)
        subway_in.columns.name = 'Station'
        subway_in.index = pd.to_datetime(subway_in.index)

        subway_in = subway_in[['Ampère Victor Hugo']] if single_station else subway_in

        # Define Invalid Dates : 
        list_of_invalid_period = []
        list_of_invalid_period.append([datetime(2019,1,10,15,30),datetime(2019,1,14,15,30)])
        list_of_invalid_period.append([datetime(2019,1,30,8,15),datetime(2019,1,30,10,30)])
        list_of_invalid_period.append([datetime(2019,2,18,11),datetime(2019,2,18,13)])
        list_of_invalid_period.append([datetime(2019,4,23,14),datetime(2019,4,28,14)])
        list_of_invalid_period.append([datetime(2019,6,26,11),datetime(2019,6,28,4)])
        list_of_invalid_period.append([datetime(2019,10,27),datetime(2019,10,28,16)])
        list_of_invalid_period.append([datetime(2019,12,21,15,45),datetime(2019,12,21,16,45)])

        invalid_dates = []
        time_step_per_hour = (60*60)/(subway_in.iloc[1].name - subway_in.iloc[0].name).seconds
        for start,end in list_of_invalid_period:
            invalid_dates = invalid_dates + list(pd.date_range(start,end,freq = f'{60/time_step_per_hour}min'))

        # Restrain invalid_dates to the df: 
        invalid_dates = list(set(invalid_dates) & set(subway_in.index))

        print(f"Time-step per hour: {time_step_per_hour}")

    elif file_name == 'Netmob.csv':
        netmob = ...

    else:
        raise NotImplementedError(f"file name option '{file_name}' has not been defined in 'load_row_data' ")
        
    return(subway_in,invalid_dates,time_step_per_hour)

def load_prediction(trainer,dataset,dataloader,args,training_mode,normalize):
    data=  [[x,y,t] for x,y,t in dataloader[training_mode]] 
    X = torch.cat([x for x,_,_ in data]).to(args.device)
    Y = torch.cat([y for _,y,_ in data]).to(args.device)
    T = torch.cat([t for _,_,t in data]).to(args.device)
    if normalize:
        Preds,Y,T = trainer.testing(dataset,False,training_mode,X, Y,T)
    else :
        Preds,Y,T = trainer.test_prediction(False,training_mode,X,Y,T)
    return(Preds,Y,T)


def get_dic_results(trainer,pi):
    results = {}
    results['MPIW'] = pi.mpiw
    results['PICP'] = pi.picp  
    results['Q'] = pi.Q_tensor.mean().item()
    results['last train loss'] = trainer.train_loss[-1]
    results['last valid loss'] = trainer.valid_loss[-1] 
    return(results)

def display_info_on_dataset(dataset,remaining_dates,train_indice = None,valid_indice = None):
    assert train_indice is not None, 'Train / Valid Split Indices has not been defined'

    if len(remaining_dates) > 0:
        train_dates1 =  f"{str(remaining_dates.iloc[0].item())}" 
        train_dates2 = f"{str(remaining_dates.iloc[train_indice].item())}"
        len_train = f"{len(remaining_dates[:train_indice])}"

        valid_dates1 =  None if (dataset.valid_prop == 0) else f"{str(remaining_dates.iloc[train_indice].item())}"
        valid_dates2 =  None if (dataset.valid_prop == 0) else f"{str(remaining_dates.iloc[valid_indice].item())}"  
        len_valid = f"{len(remaining_dates[train_indice:valid_indice])}"      

        test_dates1 =  None if ((abs(dataset.valid_prop + dataset.train_prop -1 ) < 1e-4) or (dataset.train_prop == 1 )) else f"{str(remaining_dates.iloc[valid_indice].item())}"
        test_dates2 =  None if ((abs(dataset.valid_prop + dataset.train_prop -1 ) < 1e-4) or (dataset.train_prop == 1 )) else f"{str(remaining_dates.iloc[-1].item())}"  
        len_test = f"{len(remaining_dates[valid_indice:])}"        

    else:
        train_dates1,train_dates2,valid_dates1,valid_dates2,test_dates1,test_dates2 = f"      None      ",f"      None      ",f"      None      ",f"      None      ",f"      None      ",f"      None      "
    

    print(f"Length full df: {len(dataset.df)}. \
    \n{'{:.0%}'.format(len(dataset.df_verif)/len(dataset.df))} of remaining dates after shifting lagged feature\
    \n{'{:.0%}'.format(1-len(remaining_dates)/len(dataset.df_verif))} of forbidden dates among remaining dates. \
    \n{'{:.0%}'.format(len(remaining_dates)/len(dataset.df))} of remaining sequences from Initial DataFrame \n " )

    print('              -- First Train   --   Last Train - First valid    --    Last Valid - First Test    --  Last Test')
    print(f"            {train_dates1}   --   {train_dates2 if train_dates2 == valid_dates1 else train_dates2 + ' None'}      --     {valid_dates2 if valid_dates2 == test_dates1 else valid_dates2 + ' None'}    --  {test_dates2}" ) 
    print(f" Len Train/Valid/Test:           {len_train}            ---            {len_valid}             ---             {len_test}\n")
    # ...

def get_loss(loss_function_type,args = None):
    if (loss_function_type == 'mse') or  (loss_function_type == 'MSE') or (loss_function_type == 'Mse'):
        return nn.MSELoss()
    if (loss_function_type == 'quantile') or (loss_function_type == 'Quantile') (loss_function_type == 'quantile loss') or (loss_function_type == 'QunatileLoss') or (loss_function_type == 'Qunatile Loss'):
        quantiles = torch.Tensor([args.alpha/2,1-args.alpha/2]).to(args.device)
        assert args.out_dim == len(quantiles), f"Output dimension {args.out_dim} doesn't match with the number of estimated quantiles {len(quantiles)}"
        return QuantileLoss(quantiles)

def choose_optimizer(model,args):
    # Training and Calibration :
    specific_lr = optimizer_specific_lr(model,args) if ((args.specific_lr) and (args.time_embedding)) else None

    if args.optimizer == 'adam':
        if specific_lr is not None: 
            return Adam(specific_lr,lr=args.lr,weight_decay= args.weight_decay)
        else:
            return Adam(model.parameters(),lr=args.lr,weight_decay= args.weight_decay)
    elif args.optimizer == 'sgd':
        if specific_lr is not None: 
            return SGD(specific_lr,lr=args.lr,weight_decay=args.weight_decay, momentum = args.momentum)
        else:
            return SGD(model.parameters(),lr=args.lr,weight_decay =args.weight_decay, momentum = args.momentum)
    elif args.optimizer == "adamw":
        if specific_lr is not None: 
            return AdamW(specific_lr,lr=args.lr,weight_decay= args.weight_decay)
        else:
            return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else :
        raise NotImplementedError(f'ERROR: The optimizer is not set in args or is not implemented.')
    
def load_scheduler(optimizer,args):
    if args.scheduler is None :
        scheduler = None
    else:
        scheduler1 = LinearLR(optimizer,total_iters = args.torch_scheduler_milestone, start_factor = args.torch_scheduler_lr_start_factor)
        scheduler2 = ExponentialLR(optimizer, gamma=args.torch_scheduler_gamma)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[args.torch_scheduler_milestone])
    return(scheduler)

def load_init_trainer(folder_path,file_name,args):
    # Load dataset and invalid_dates 
    dataset,invalid_dates = get_DataSet_and_invalid_dates(args.abs_path,folder_path,file_name,args.W,args.D,args.H,args.step_ahead,single_station = args.single_station)
    (Datasets,DataLoader_list,time_slots_labels,dic_class2rpz,dic_rpz2class,nb_words_embedding) = dataset.split_K_fold(args,invalid_dates)
    return(Datasets,DataLoader_list,dic_class2rpz,nb_words_embedding,time_slots_labels,dic_rpz2class)
