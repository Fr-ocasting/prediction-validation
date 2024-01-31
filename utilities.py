
import pandas as pd
import torch
import numpy as np
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

class DataSet(object):
    def __init__(self,df,init_df = None,mini= None, maxi = None, mean = None, normalized = False,time_step_per_hour = None):
        self.length = len(df)
        self.df = df
        self.normalized = normalized
        self.time_step_per_hour = time_step_per_hour
        self.df_dates = pd.DataFrame(self.df.index,index = np.arange(len(self.df)),columns = ['date'])
        if time_step_per_hour is not None :
            self.Week_nb_steps = int(7*24*self.time_step_per_hour)
            self.Day_nb_steps = int(24*self.time_step_per_hour)
        else : 
            self.Week_nb_steps = None
            self.Day_nb_steps = None

        if mini is not None: 
            self.mini = mini
        else : 
            self.mini = df.min()

        if maxi is not None: 
            self.maxi = maxi
        else : 
            self.maxi = df.max()

        if mean is not None:
            self.mean = mean
        else:
            self.mean = df.mean()

        if init_df is not None:
            self.init_df = init_df
        else:
            self.init_df = df
        
    def normalize(self):
        if self.normalized:
            print('The df might be already normalized')
        minmaxnorm = lambda x : (x-self.mini)/(self.maxi-self.mini)
        normalized_df = minmaxnorm(self.init_df)
        normalized_Xt = DataSet(normalized_df,init_df = self.init_df,mini = self.mini, maxi = self.maxi, mean = self.mean,normalized=True,time_step_per_hour=self.time_step_per_hour)
        return(normalized_Xt)
    
    def unormalize(self,timeserie):
        if not(self.normalized):
            print('The df might be already unormalized')
        return(timeserie*(self.maxi - self.mini)+self.mini)
    
    def get_time_serie(self,station):
        timeserie = TimeSerie(ts = self.df[[station]],init_ts = self.init_df[[station]],mini = self.mini[station],maxi = self.maxi[station],mean = self.mean[station], normalized = self.normalized)
        return(timeserie)

    def shift_data(self,step_ahead,historical_len,Weeks,Days):

        # Weekkly periodic
        Uwt = [torch.unsqueeze(torch.Tensor(self.df.shift((Weeks-i)*self.Week_nb_steps).values),2) for i in range(Weeks)]
        Dwt = [self.df_dates.shift((Weeks-i)*self.Week_nb_steps) for i in range(Weeks)] 

        # Daily periodic
        Udt = [torch.unsqueeze(torch.Tensor(self.df.shift((Days-i)*self.Day_nb_steps).values),2) for i in range(Days)]
        Ddt = [self.df_dates.shift((Days-i)*self.Day_nb_steps) for i in range(Days)] 

        # Recent Historic pattern 
        Ut =  [torch.unsqueeze(torch.Tensor(self.df.shift(step_ahead+(historical_len-i)).values),2) for i in range(1,historical_len+1)]
        Dt = [self.df_dates.shift(step_ahead+(historical_len-i)) for i in range(1,historical_len+1)] 

        shifted_values = Uwt+Udt+Ut
        shifted_dates = Dwt+Ddt+Dt

        return(shifted_values,shifted_dates)

    def get_feature_vect(self,step_ahead,historical_len,Days,Weeks):
        if self.time_step_per_hour is None :
            raise Exception('Number of time steps per hour as not been defined. Please use FeatureVector.time_step_per_hour ')
        
        else : 
            # Get the shifted "Dates" of Feature Vector and Target
            (shifted_values,shifted_dates) = self.shift_data(step_ahead,historical_len,Weeks,Days)
            L_shifted_dates = shifted_dates + [self.df_dates]
            Names = [f't-{str(self.Week_nb_steps*(Weeks-w))}' for w in range(Weeks)] + [f't-{str(self.Day_nb_steps*(Days-d))}' for d in range(Days)] + [f't-{str(historical_len-t)}' for t in range(historical_len)]+ ['t']
            df_verif = pd.DataFrame({name:lst['date'] for name,lst in zip(Names,L_shifted_dates)})[self.Week_nb_steps+Weeks-1:]

            # Get Feature Vector and Target 
            U = torch.cat(shifted_values,dim=2)[:][self.Week_nb_steps+Weeks-1:]
            Utarget = torch.unsqueeze(torch.Tensor(self.df.values),2)[self.Week_nb_steps+Weeks-1:]
            return(U,Utarget,df_verif)
    

class TimeSerie(object):
    def __init__(self,ts,init_ts = None,mini = None, maxi = None, mean = None, normalized = False):
        self.length = len(ts)
        self.ts = ts
        self.normalized = normalized
        if mini is not None: 
            self.mini = mini
        else : 
            self.mini = ts.min()
        if maxi is not None: 
            self.maxi = maxi
        else : 
            self.maxi = ts.max()
        if mean is not None:
            self.mean = mean
        else:
            self.mean = ts.mean()
        if init_ts is not None:
            self.init_ts = init_ts
        else:
            self.init_ts = ts
        
    def normalize(self):
        if self.normalized:
            print('The TimeSerie might be already normalized')
        minmaxnorm = lambda x : (x-self.mini)/(self.maxi-self.mini)
        return(minmaxnorm(self.init_ts))
    
    def unormalize(self):
        if not(self.normalized):
            print('The TimeSerie might be already unnormalized')
        return(self.ts*(self.maxi - self.mini)+self.mini)