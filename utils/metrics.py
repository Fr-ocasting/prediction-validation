import torch.nn as nn
import torch


def evaluate_metrics(Preds,Y_true,metrics = ['mse','mae','mape','mse by station','mae by station','mape by station']):
    dic_metric = {}
    for metric in metrics :
        if 'by station' in metric:
            metric_name = metric.split(' ')[0]
            error = metrics_by_station(Preds,Y_true,metric_name)
        else:
            fun = load_fun(metric)
            error = fun(Preds,Y_true).item()

        dic_metric[metric] = error
    return(dic_metric)

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


def personnal_MAPE(Preds,Y_true):
    Y_true = Y_true.reshape(-1)
    Preds = Preds.reshape(-1)
    mask = Y_true>0
    error = 100*torch.mean(torch.abs(Y_true[mask] - Preds[mask])/Y_true[mask])
    return error




