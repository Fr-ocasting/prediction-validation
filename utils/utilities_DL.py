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
    from torch.optim.lr_scheduler import LinearLR,ExponentialLR,SequentialLR,MultiStepLR
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
from constants.config import optimizer_specific_lr, convert_into_parameters
from DL_class import QuantileLoss
from dataset import  DataSet
from TE_transfer_learning import TE_transfer
from dl_models.full_model import full_model
from utils.losses import masked_mse, masked_mae, masked_rmse, masked_mape


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
    
    # Specific LR for TimeEmbedding : 
    args_embedding = args.args_embedding
    if ('calendar_embedding' in args.dataset_names) and (hasattr(args_embedding,'specific_lr')) and (args_embedding.specific_lr)>0: 
        specific_lr = optimizer_specific_lr(model,args)
    else:
        specific_lr = None

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    if args.optimizer in ['adam','Adam']:
        if specific_lr is not None: 
            return Adam(specific_lr,lr=args.lr,weight_decay= args.weight_decay)
        else:
            return Adam(model_parameters,lr=args.lr,weight_decay= args.weight_decay)
    elif args.optimizer == 'sgd':
        if specific_lr is not None: 
            return SGD(specific_lr,lr=args.lr,weight_decay=args.weight_decay, momentum = args.momentum)
        else:
            return SGD(model_parameters,lr=args.lr,weight_decay =args.weight_decay, momentum = args.momentum)
    elif args.optimizer == "adamw":
        if specific_lr is not None: 
            return AdamW(specific_lr,lr=args.lr,weight_decay= args.weight_decay)
        else:
            return AdamW(model_parameters, lr=args.lr, weight_decay=args.weight_decay)
    else :
        raise NotImplementedError(f'ERROR: The optimizer is not set in args or is not implemented.')

def load_scheduler(optimizer,args):
    if (args.scheduler is None) or (math.isnan(args.scheduler)) or (args.scheduler == False):
        scheduler = None
    else:
        
        if args.torch_scheduler_type == 'MultiStepLR':
            scheduler = MultiStepLR(optimizer, milestones=args.torch_scheduler_milestone, gamma=args.torch_scheduler_gamma)
        elif args.torch_scheduler_type == 'warmup':
            scheduler1 = LinearLR(optimizer,total_iters = args.torch_scheduler_milestone, start_factor = args.torch_scheduler_lr_start_factor)
            scheduler2 = ExponentialLR(optimizer, gamma=args.torch_scheduler_gamma)
            scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[args.torch_scheduler_milestone])
        else:
            raise NotImplementedError(f'ERROR: The scheduler is not set in args or is not implemented. Please check the args.torch_scheduler_type. Set to MultiStepLR or warmup.')
    return(scheduler)


def get_loss(args):
    if (args.loss_function_type == 'mse') or  (args.loss_function_type == 'MSE') or (args.loss_function_type == 'Mse'):
        return nn.MSELoss()
    elif (args.loss_function_type == 'mae') or (args.loss_function_type == 'MAE') or (args.loss_function_type == 'Mae'):
        return nn.L1Loss()
    
    elif (args.loss_function_type == 'masked_mse') or (args.loss_function_type == 'MaskedMSE') or (args.loss_function_type == 'Masked MSE'):
        return masked_mse
    elif (args.loss_function_type == 'masked_mae') or (args.loss_function_type == 'MaskedMAE') or (args.loss_function_type == 'Masked MAE'):
        return masked_mae
    elif (args.loss_function_type == 'masked_rmse') or (args.loss_function_type == 'MaskedRMSE') or (args.loss_function_type == 'Masked RMSE'):
        return masked_rmse
    elif (args.loss_function_type == 'masked_mape') or (args.loss_function_type == 'MaskedMAPE') or (args.loss_function_type == 'Masked MAPE'):
        return masked_mape
    
    elif (args.loss_function_type == 'huber') or (args.loss_function_type == 'Huber') or (args.loss_function_type == 'HuberLoss'):
        return nn.HuberLoss()
    elif (args.loss_function_type == 'masked_huber') or (args.loss_function_type == 'MaskedHuber') or (args.loss_function_type == 'Masked Huber'):
        return NotImplementedError
    
    elif (args.loss_function_type == 'quantile') or (args.loss_function_type == 'Quantile') (args.loss_function_type == 'quantile loss') or (args.loss_function_type == 'QunatileLoss') or (args.loss_function_type == 'Qunatile Loss'):
        quantiles = torch.Tensor([args.alpha/2,1-args.alpha/2]).to(args.device)
        assert args.out_dim == len(quantiles), f"Output dimension {args.out_dim} doesn't match with the number of estimated quantiles {len(quantiles)}"
        return QuantileLoss(quantiles)
    else: 
        raise NotImplementedError(f'ERROR: The loss function is not set in args or is not implemented. Please check the args.loss_function_type = {args.loss_function_type}.')
# ============================================
# USELESS. HAS TO BE REMOVED
# ============================================


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