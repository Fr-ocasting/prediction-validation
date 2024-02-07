
import pandas as pd
import torch
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def str2neg(x):
    # Convert string values into '-1'. Keep the nan values
    try:
        return(int(x))
    except:
        if x is np.nan :
            return(x)
        else:
          return(-1)

def str_xa2int(x):
    # Convert string values with a space '\xa0' into int values
    try:
        return(int(x.replace('\xa0','')))
    except:
        return(x)
    


def date_from_month_offset(n):
    n = int(n)
    reference_date = datetime(2019, 10, 1)
    years, months = divmod(n, 12)
    result_date = reference_date + timedelta(days=365 * years) + relativedelta(months=months)
    return f'{result_date.date().month}-{result_date.date().year}'

def get_mode_date2path(df_list,df_names):
    ''' Return a dic which associate a Mode and a date to the csv_path'''
    dic = {}
    for list_paths,mode in zip(df_list,df_names):
        dic[mode] = {}
        for csv_path in list_paths:
            csv_path = csv_path.replace('\\','/')
            month = csv_path.split('/')[2].split('_')[0]
            dic[mode][date_from_month_offset(month)] = csv_path 
    return(dic)

# Commit innutile 
def get_batch(X,Y,batch_size,shuffle = True):
    n = X.shape[0]
    if len(X.shape) < 3:
        X = X.reshape(X.shape[0],1,X.shape[1])
    indices = np.arange(n)
    if shuffle:
        random.shuffle(indices)
    for idx in range(0,n,batch_size):
        yield X[indices[idx:min(idx+batch_size,n)]], Y[indices[idx:min(idx+batch_size,n)]]

def unormalize_tensor(tensor,mini,maxi):
    mini = torch.Tensor(mini.values).unsqueeze(1)
    maxi = torch.Tensor(maxi.values).unsqueeze(1)   
    return tensor*(maxi- mini)+mini

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