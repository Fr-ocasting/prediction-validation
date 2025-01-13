import torch.nn as nn
import torch
import sys
import os
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from PI.PI_object import PI_object

def evaluate_metrics(Preds,Y_true,metrics, alpha = None, type_calib = None,dic_metric = {}):
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
    if ('PICP' in metrics) or ('MPIW' in metrics): 
        dic_metric,metrics  = evaluate_PI(dic_metric,Preds,Y_true,alpha,type_calib,metrics)

    dic_metric  = evaluate_single_point_metrics(dic_metric,Preds,Y_true,metrics)
    return(dic_metric)

def evaluate_single_point_metrics(dic_metric,Preds,Y_true,metrics):
    '''
    Tackle the case of single point prediction
    '''
    for metric in metrics :
        if 'by station' in metric:
            metric_name = metric.split(' ')[0]
            error = metrics_by_station(Preds,Y_true,metric_name)
        else:
            fun = load_fun(metric)
            error = fun(Preds,Y_true).item()
        dic_metric[metric] = error
    return dic_metric

def evaluate_PI(dic_metric,Preds,Y_true,alpha,type_calib,metrics):
    '''
    Tackle the case of multi-point (range) prediction
    '''    
    pi = PI_object(Preds,Y_true,alpha,type_calib)
    for metric in metrics:
        if metric == 'PICP':
            dic_metric[metric] = pi.picp
        if metric == 'MPIW':
            dic_metric[metric] = pi.mpiw
    metrics = [metric for metric in metrics if metric not in ['PICP','MPIW']]

    return(dic_metric,metrics)

def load_fun(metric):
    if metric == 'mse':
        fun = nn.MSELoss()
    if metric == 'mae':
        fun = nn.L1Loss()
    if metric == 'mape':
        fun = personnal_MAPE
    return(fun)

def metrics_by_station(Preds,Y_true,metric_name):
    fun = load_fun(metric_name)
    errors = []
    for station in range(Y_true.size(1)):
        P = Preds[:,station,:]
        Y = Y_true[:,station,:]
        error = fun(P,Y)
        errors.append(error.item())
    return(errors) 


def personnal_MAPE(Preds,Y_true,inf_border=0):
    '''
    args
    -------
    inf_border: compute MAPE only when expected values > inf_border. 
    '''
    Y_true = Y_true.reshape(-1)
    Preds = Preds.reshape(-1)
    mask = Y_true>inf_border
    error = 100*torch.mean(torch.abs(Y_true[mask] - Preds[mask])/Y_true[mask])
    return error

def error_along_ts(predict,real,metric,min_flow,normalize):
       if not isinstance(real, torch.Tensor):
        real = torch.tensor(real.values).reshape(-1)
        predict = torch.tensor(predict.values).reshape(-1)
       else:
        real = real.detach().clone().reshape(-1)
        predict = predict.detach().clone().reshape(-1)            

       mask = real>min_flow
       error = torch.full(real.shape, -1.0)  # Remplir avec -1 par défaut
       if metric == 'mape':
              error[mask] = 100 * (torch.abs(real[mask] - predict[mask]) / real[mask]) 
              error[mask] = torch.clamp(error[mask], max=100)

       elif metric == 'mae':
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
              raise NotImplementedError
       
       return(error)




