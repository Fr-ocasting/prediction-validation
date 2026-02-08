import torch.nn as nn
import torch
import sys
import os
import pandas as pd 
import numpy as np
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from pipeline.PI.PI_object import PI_object
from pipeline.utils.losses import masked_mse, masked_mae, masked_rmse, masked_mape,RMSELoss

def evaluate_metrics(Preds,Y_true,metrics, alpha = None, type_calib = None,dic_metric = None,previous=None, horizon_step = None,Q = None):
    '''
    Args:
    ------
    Preds : torch.Tensor(). Preds.size(-1) is the output dim of the model. 
    >> If Preds.size(-1) == 1, then single-point prediction 
    >> If Preds.size(-1) == 2, then quantile regression  

    Y_true : torch.Tensor(). is the True value of the future data. 

    Metrics : name of metrics which are evaluated.
    >> Can be set to evaluate a mean value through an entire tensor  ('mse','mae','mape')
    >> Can be set to evaluate a mean value by spatial unit ('mse by station','mae by station','mape by station')
    >> Can be set to evaluate a PI  ('PICP','MPIW')
    >> choices of metrics : = ['mse','mae','mape','mse by station','mae by station','mape by station','PICP','MPIW']

    Alpha: parameter of the quantile regression, represent the expected 1- alpha/2 th predicted quantile

    type_calib: can be set as 'CQR' (for Conformalized Quantile Regression) or 'classic' (for Quantile Regression). 

    dic_metric : initialization of the output

    Output:
    -------
    Return a dictionnary (dic_metric) of couple (Key, error), where the key are the metrics, and the values the associated error from the prediction.
    '''
    if dic_metric is None:
        dic_metric = {}
    if ('PICP' in metrics) or ('MPIW' in metrics): 
        dic_metric,metrics  = evaluate_PI(dic_metric,Preds,Y_true,alpha,type_calib,metrics,Q = Q)

    dic_metric  = evaluate_single_point_metrics(dic_metric,Preds,Y_true,metrics,previous,horizon_step)
    return(dic_metric)

def evaluate_single_point_metrics(dic_metric,Preds,Y_true,metrics,previous,horizon_step):
    '''
    Tackle the case of single point prediction
    '''

    for out_dim in range(Preds.size(-1)):
        Preds_i = Preds[...,out_dim]
        Y_true_i = Y_true[...,out_dim]
        for metric in metrics :
            if 'by station' in metric:
                metric_name = metric.split(' ')[0]
                error = metrics_by_station(Preds_i,Y_true_i,metric_name,previous=previous)
            else:
                fun = load_fun(metric,previous=previous)
                error = fun(Preds_i,Y_true_i).item()
            dic_metric[f"{metric}_h{(out_dim+1)*horizon_step}"] = error

    for metric in metrics:
        dic_metric[f"{metric}_all"] = np.mean(np.array([dic_metric[f"{metric}_h{(out_dim+1)*horizon_step}"] for out_dim in range(Preds.size(-1))]))
    return dic_metric

def evaluate_PI(dic_metric,Preds,Y_true,alpha,type_calib,metrics,Q = None):
    '''
    Tackle the case of multi-point (range) prediction
    '''    
    pi = PI_object(Preds,Y_true,alpha,type_calib,Q = Q)
    for metric in metrics:
        if metric == 'PICP':
            dic_metric[metric] = pi.picp
        if metric == 'MPIW':
            dic_metric[metric] = pi.mpiw
    metrics = [metric for metric in metrics if metric not in ['PICP','MPIW']]

    return(dic_metric,metrics)

def load_fun(metric,previous=None):
    if metric == 'mse':
        fun = nn.MSELoss()
    elif metric == 'rmse':
        fun = RMSELoss()
    elif metric == 'mae':
        fun = nn.L1Loss()
    elif metric == 'mape':
        fun = personnal_MAPE
    elif metric == 'mase':
        def fun(Preds,Y_true):
            return personnal_MASE(Preds,Y_true,previous = previous)
        
    elif metric == 'masked_mse':
        fun = masked_mse
    elif metric == 'masked_mae':
        fun = masked_mae
    elif metric == 'masked_mape':
        fun = masked_mape
    elif metric == 'masked_rmse': 
        fun = masked_rmse
    else:
        raise NotImplementedError(f"Metric {metric} is not implemented.")
    return(fun)

def metrics_by_station(Preds,Y_true,metric_name,previous=None):
    fun = load_fun(metric_name,previous=previous)
    errors = []
    for station in range(Y_true.size(1)):
        P = Preds[:,station,:]
        Y = Y_true[:,station,:]
        error = fun(P,Y)
        errors.append(error.item())
    return(errors) 


def personnal_MASE(Preds,Y_true,previous = None):
    '''
    args
    -------
    '''
    if previous is None:
        previous = Y_true[:-1]
        Y_true = Y_true[1:]
        Preds = Preds[1:]

    MAE_naiv = torch.mean(torch.abs(Y_true-previous))
    MAE = torch.mean(torch.abs(Y_true-Preds))

    error = MAE/MAE_naiv
    return error

def personnal_MAPE(Preds,Y_true,inf_border=0):
    '''
    args
    -------
    inf_border: compute MAPE only when expected values > inf_border. 
    '''
    Y_true = Y_true.reshape(-1)
    Preds = Preds.reshape(-1)
    mask = Y_true>(inf_border+1e-3)
    #print('Y_true: ',pd.DataFrame(Y_true.view(-1)).describe())
    #print('Preds: ',pd.DataFrame(Preds.view(-1)).describe())

    error = 100*torch.mean(torch.abs(Y_true[mask] - Preds[mask])/Y_true[mask])
    return error

def error_along_ts(predict,real,metric,min_flow,normalize):
       if not isinstance(real, torch.Tensor):
        real = torch.tensor(real.values).reshape(-1)
        predict = torch.tensor(predict.values).reshape(-1)
       else:
        real = real.detach().clone().reshape(-1)
        predict = predict.detach().clone().reshape(-1)            

       mask = real>min_flow if min_flow is not None else torch.ones_like(real).bool()
       error = torch.full(real.shape, -1.0)  # Remplir avec -1 par défaut
       if metric == 'mape':
              error[mask] = 100 * (torch.abs(real[mask] - predict[mask]) / real[mask]) 
              error[mask] = torch.clamp(error[mask], max=100)

       elif (metric == 'mae') or (metric == 'rmse'):
              err = torch.abs(real[mask] - predict[mask])
              if normalize:
                  err = 100 * err/err.max()
              error[mask] = err

       elif metric == 'mse':
              err = (real[mask] - predict[mask])**2
              if normalize:
                  err = 100 * err/err.max()
              error[mask] = err

       else:
              raise NotImplementedError(f"Metric {metric} is not implemented for error_along_ts function.")
       
       return(error)


