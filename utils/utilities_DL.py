import numpy as np
import torch 
import torch.nn as nn 
import pandas as pd
from datetime import datetime
import argparse
import inspect
import math 
from torch.optim import SGD,Adam,AdamW
try:
    from torch.optim.lr_scheduler import LinearLR,ExponentialLR,SequentialLR
except:
    print(f'Pytorch version {torch.__version__} does not allow you to use lr-scheduler')

# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(current_file_path,'..'))
if ROOT not in sys.path:
    sys.path.insert(0,ROOT)
# ...

# Personnal import: 
from constants.config import optimizer_specific_lr, get_config_embed, convert_into_parameters
from DL_class import QuantileLoss
from dataset import  DataSet
from TE_transfer_learning import TE_transfer
from dl_models.full_model import full_model


def get_args_embedding(args,nb_words_embedding):
    if args.time_embedding:
        config_Tembed = get_config_embed(nb_words_embedding,embedding_dim = args.embedding_dim,position = args.position)
        args_embedding = convert_into_parameters(config_Tembed)
    else:
        args_embedding = argparse.ArgumentParser(description='TimeEmbedding').parse_args(args=[])
    return(args_embedding)


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

def get_DataSet_and_invalid_dates(W,D,H,step_ahead,dataset_names,single_station = False,coverage_period = None):
    df,invalid_dates,time_step_per_hour = load_raw_data(dataset_names,single_station = single_station)
    if coverage_period is not None:
        df = df.loc[coverage_period]
        invalid_dates = list(set(invalid_dates) & set(coverage_period))
    dataset = DataSet(df,time_step_per_hour=time_step_per_hour, Weeks = W, Days = D, historical_len= H,step_ahead=step_ahead)
    print(f"coverage period: {df.index.min()} - {df.index.max()}")
    return(dataset,invalid_dates)

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

def load_model_and_optimizer(args,dic_class2rpz):
    model =  full_model(args,dic_class2rpz).to(args.device)
    # Config optimizer:
    optimizer = choose_optimizer(model,args)
    scheduler = load_scheduler(optimizer,args)

    print('number of total parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    print('number of trainable parameters: {}'.format(sum([p.numel() for p in model.parameters() if p.requires_grad])))
    return(model,optimizer,scheduler)


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
    if (args.scheduler is None) or (math.isnan(args.scheduler)) :
        scheduler = None
    else:
        scheduler1 = LinearLR(optimizer,total_iters = args.torch_scheduler_milestone, start_factor = args.torch_scheduler_lr_start_factor)
        scheduler2 = ExponentialLR(optimizer, gamma=args.torch_scheduler_gamma)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[args.torch_scheduler_milestone])
    return(scheduler)

def get_loss(args):
    if (args.loss_function_type == 'mse') or  (args.loss_function_type == 'MSE') or (args.loss_function_type == 'Mse'):
        return nn.MSELoss()
    if (args.loss_function_type == 'quantile') or (args.loss_function_type == 'Quantile') (args.loss_function_type == 'quantile loss') or (args.loss_function_type == 'QunatileLoss') or (args.loss_function_type == 'Qunatile Loss'):
        quantiles = torch.Tensor([args.alpha/2,1-args.alpha/2]).to(args.device)
        assert args.out_dim == len(quantiles), f"Output dimension {args.out_dim} doesn't match with the number of estimated quantiles {len(quantiles)}"
        return QuantileLoss(quantiles)
# ============================================
# USELESS. HAS TO BE REMOVED
# ============================================

def load_init_trainer(args):
    # Load dataset and invalid_dates 
    dataset,invalid_dates = get_DataSet_and_invalid_dates(args.W,args.D,args.H,args.step_ahead,single_station = args.single_station)
    (Datasets,DataLoader_list,time_slots_labels,dic_class2rpz,dic_rpz2class,nb_words_embedding) = dataset.split_K_fold(args,invalid_dates)
    return(Datasets,DataLoader_list,dic_class2rpz,nb_words_embedding,time_slots_labels,dic_rpz2class)

def get_MultiModel_loss_args_emb_opts(args,nb_words_embedding,dic_class2rpz,n_vertex = None):
    args.n_vertex = n_vertex
    loss_function = get_loss(args)
    args_embedding = get_args_embedding(args,nb_words_embedding)

    model_opt_sched_list = [load_model_and_optimizer(args,dic_class2rpz) for _ in range(args.K_fold)]
    Model_list = [model_opt[0] for model_opt in model_opt_sched_list]
    if args.TE_transfer:
        if os.path.exists(f'{args.abs_path}data/Trained_Time_Embedding{args.embedding_dim}.pkl'):
            Model_list = [TE_transfer(model,n_vertex,args,model_dir = 'data/') for model in Model_list]
        else:
            print('TE impossible')
                                      
        
    Optimizer_list = [model_opt[1] for model_opt in model_opt_sched_list]
    Scheduler_list = [model_opt[2] for model_opt in model_opt_sched_list]
    return(loss_function,Model_list,Optimizer_list,Scheduler_list,args_embedding)

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