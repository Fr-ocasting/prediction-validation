import torch.nn as nn

def evaluate_metrics(Pred,Y_true,metrics = ['mse','mae']):
    dic_metric = {}
    for metric in metrics :
        if metric == 'mse':
            fun = nn.MSELoss()
        if metric == 'mae':
            fun = nn.L1Loss()

        error = fun(Pred,Y_true)
        dic_metric[metric] = error
    return(dic_metric)