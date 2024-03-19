
import pandas as pd
import torch
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import torch.nn as nn
from scipy.spatial.distance import cdist 



def get_higher_quantile(conformity_scores,quantile_order,device = 'cpu'):
    assert 0 <= quantile_order <= 1, "Quantile order must be <= 1 and >=0"
    sorted,_ = torch.sort(conformity_scores,0)

    index = torch.ceil(torch.Tensor([conformity_scores.size(0) * quantile_order])).long() - 1
    index = torch.clamp(index, 0, conformity_scores.size(0) - 1)  # 0 <= index < len(conformity_scores) 
    return(sorted[index:index+1,:,:].to(device)) 

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
    
def get_distance_matrix(centroids1,centroids2, inv = True):
    ''' return the distance matrix, and even the inverse if we need it too '''
    dist_matrix = cdist(centroids1, centroids2, metric='euclidean')
    dist_matrix = dist_matrix/1e3

    if inv :
        sigma = np.std(dist_matrix)
        matrix = np.empty((dist_matrix.shape[0],dist_matrix.shape[1]))
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if dist_matrix[i,j] != 0 :
                    matrix[i,j] = np.exp(-(dist_matrix[i,j]/sigma)**2)
                else : 
                    matrix[i,j] = 1
    else : 
        matrix = dist_matrix
    return(matrix)


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