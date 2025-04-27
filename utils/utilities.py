
import pandas as pd
import torch
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import torch.nn as nn
from scipy.spatial.distance import cdist 
import pickle 
import io 
import inspect

def restrain_df_to_specific_period(df,coverage_period):
    if coverage_period is not None:
        df = df.loc[coverage_period]

    df = df.sort_index()
    return df

def load_inputs_from_dataloader(dataloader,device):
        inputs_i = [[x,y,x_c] for  x,y,x_c  in dataloader]
        X = torch.cat([x for x,_,_ in inputs_i]).to(device)
        Y = torch.cat([y for _,y,_ in inputs_i]).to(device)
        nb_contextual = len(next(iter(dataloader))[2])
        X_c = [torch.cat([x_c[k] for _,_,x_c in inputs_i]).to(device) for k in range(nb_contextual)]
        return X,Y,X_c,nb_contextual

def filter_args(func, args):
    sig = inspect.signature(func)
    #valid_args = {k: v for k, v in args.items() if k in sig.parameters}
    filered_args = {k: v for k, v in vars(args).items() if k in sig.parameters}
    return filered_args

def get_higher_quantile(conformity_scores,quantile_order,device = 'cpu'):
    assert 0 <= quantile_order <= 1, "Quantile order must be <= 1 and >=0"
    sorted,_ = torch.sort(conformity_scores,0)

    index = torch.ceil(torch.Tensor([conformity_scores.size(0) * quantile_order])).long() - 1
    index = torch.clamp(index, 0, conformity_scores.size(0) - 1)  # 0 <= index < len(conformity_scores) 
    return(sorted[index:index+1,:,:].to(device)) 


def get_INVALID_DATES(list_of_invalid_period,freq):
    INVALID_DATES = []
    for start,end in list_of_invalid_period:
        INVALID_DATES = INVALID_DATES + list(pd.date_range(start,end,freq = freq))
    return INVALID_DATES

def get_time_step_per_hour(freq):
    if 'min' in freq:
        freq_i = int(freq.split('min')[0])
        return 60/freq_i
    elif ('h' in freq):
        freq_i = int(freq.split('h')[0])
        return 1/freq_i
    elif ('H' in freq):
        freq_i = int(freq.split('H')[0])
        return 1/freq_i
    elif ('d' in freq): 
        freq_i = int(freq.split('d')[0])
        return 1/(freq_i*24)      
    elif ('D' in freq): 
        freq_i = int(freq.split('D')[0])
        return 1/(freq_i*24)         
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
            month = csv_path.split('/')[-1].split('_')[0]
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

def get_holidays(year):
    holidays = []
    # New Year's Day
    holidays.append(datetime(year=year, month=1, day=1))
    # Easter Sunday
    if year == 2019:
        day,month = 21,4
    if year == 2020:
        day,month = 13,4
    if year == 2021:
        day,month = 5,4  
    if year == 2022:
        day,month = 18,4
    holidays.append(datetime(year=year, month=month, day=day))
    # Labor Day
    holidays.append(datetime(year=year, month=5, day=1))
    # Victory in Europe Day
    holidays.append(datetime(year=year, month=5, day=8))
    # Ascension 
    if year == 2019:
        day,month = 30,5
    if year == 2020:
        day,month = 21,5
    if year == 2021:
        day,month = 13,5  
    if year == 2022:
        day,month = 26,5    
    holidays.append(datetime(year=year, month=month, day=day))
    # Bastille Day
    holidays.append(datetime(year=year, month=7, day=14))
    #"Pentecôte" 's Monday
    if year == 2019:
        day,month = 10,6
    if year == 2020:
        day,month = 1,6
    if year == 2021:
        day,month = 24,5 
    if year == 2022:
        day,month = 6,6    
    holidays.append(datetime(year=year, month=month, day=day))
    # Assumption of Mary
    holidays.append(datetime(year=year, month=8, day=15))
    # All Saints' Day
    holidays.append(datetime(year=year, month=11, day=1))
    # Armistice Day
    holidays.append(datetime(year=year, month=11, day=11))
    # Christmas Day
    holidays.append(datetime(year=year, month=12, day=25))
    
    #holidays = [holiday.timestamp() for holiday in holidays]
    return holidays

def get_school_holidays(city,freq = '15min'):
    print('Only French Holidays from 2019 to 2020 has been implemented')
    if city == 'Lyon':
        winter_holidays = list(pd.date_range(start = pd.to_datetime('16/02/2019',dayfirst=True), end = pd.to_datetime('4/03/2019',dayfirst=True), freq = freq)) + list(pd.date_range(start = pd.to_datetime('22/02/2020',dayfirst=True), end = pd.to_datetime('9/03/2020',dayfirst=True), freq = freq)) 
        spring_holidays = list(pd.date_range(start = pd.to_datetime('13/04/2019',dayfirst=True), end = pd.to_datetime('29/04/2019',dayfirst=True), freq = freq)) + list(pd.date_range(start = pd.to_datetime('18/04/2020',dayfirst=True), end = pd.to_datetime('4/05/2020',dayfirst=True), freq = freq)) 
        summer_holidays = list(pd.date_range(start = pd.to_datetime('6/07/2019',dayfirst=True), end = pd.to_datetime('02/09/2019',dayfirst=True), freq = freq))  + list(pd.date_range(start = pd.to_datetime('04/07/2020',dayfirst=True), end = pd.to_datetime('31/08/2020',dayfirst=True), freq = freq)) 
        autumn_holidays = list(pd.date_range(start = pd.to_datetime('19/10/2019',dayfirst=True), end = pd.to_datetime('04/11/2019',dayfirst=True), freq = freq))
        christmas_holidays = list(pd.date_range(start = pd.to_datetime('21/12/2019',dayfirst=True), end = pd.to_datetime('06/01/2020',dayfirst=True), freq = freq))
    else:
        raise NotImplementedError(f'City of {city} has not been implemented')

    school_holidays = winter_holidays+spring_holidays+summer_holidays+autumn_holidays+christmas_holidays
    return(school_holidays)



def get_time_delta_holidays(agg_minutes = True,agg_hour = False):
    holidays = []
    for year in [2019,2020] :
        holidays = holidays+ get_holidays(year)
    if agg_minutes:
        holidays = [[holiday + k*timedelta(minutes = 15) for k in range(24*4)] for holiday in holidays]
    elif agg_hour: 
        holidays = [[holiday + k*timedelta(hour = 1) for k in range(24)] for holiday in holidays]
    holidays = list(np.concatenate(holidays))
    return(holidays)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)