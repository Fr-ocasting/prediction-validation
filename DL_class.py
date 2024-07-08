import pandas as pd 
import numpy as np
import torch 
import torch.nn as nn 


class QuantileLoss(nn.Module):
    def __init__(self,quantiles):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        # y-^y 
        errors = target - preds       #Soustraction sur la dernière dimension, à priori target 1 sortie et prediction len(quantiles) sorties

        # Errors : [B,N,2]  cause target [B,N,1] and preds [B,N,2]  
        losses = torch.max(self.quantiles*errors,(self.quantiles-1)*errors) # Récupère le plus grand des deux écart, pour chacune des estimations de quantile
        
        # Prends la moyenne de toute les erreurs
        loss = torch.mean(torch.sum(losses,dim = -1))   #  Loss commune pour toutes les stations. sinon loss par stations : torch.mean(torch.sum(losses,dim = -1),dim = 0)

        return(loss)


class TrainValidTest_Split_Normalize(object):
    def __init__(self,data,dims,
                 train_indices = None , valid_indices = None, test_indices = None,
                 first_train = None, last_train = None, first_valid = None, last_valid = None, first_test = None, last_test = None, 
                 minmaxnorm = False,standardize = False):
        super(TrainValidTest_Split_Normalize,self).__init__()
        self.data = data
        self.minmaxnorm = minmaxnorm
        self.standardize = standardize
        self.dims = dims 

        if train_indices is not None : self.train_indices = train_indices
        if valid_indices is not None : self.valid_indices = valid_indices
        if test_indices is not None : self.test_indices = test_indices

        if first_train is not None : self.first_train = first_train
        if last_train is not None : self.last_train = last_train

        if first_valid is not None : self.first_valid = first_valid
        if last_valid is not None : self.last_valid = last_valid

        if first_test is not None : self.first_test = first_test
        if last_test is not None : self.last_test = last_test

        self.split_data()

    def split_data(self):
        # Split Data within 3 groups:
        if hasattr(self,'train_indices'):
            self.data_train = self.data[self.train_indices] 
            self.data_valid = self.data[self.valid_indices] if self.valid_indices is not None else None
            self.data_test = self.data[self.test_indices] if self.test_indices is not None else None
        elif hasattr(self,'first_train'):
            self.data_train = self.data[self.first_train:self.last_train]
            self.data_valid = self.data[self.first_valid:self.last_valid] if self.first_valid is not None else None
            self.data_test = self.data[self.first_test:self.last_test]   if self.first_test is not None else None
        else: 
            raise ValueError("Neither 'train_indices' nor 'first_train' attribute has been designed ")
        

    def load_normalize_tensor_datasets(self,mini = None, maxi = None, mean = None, std = None):
        '''Load TensorDataset (train_dataset) object from data_train.
        Define TensorDataset object from valid (valid_dataset) and test (test_dataset). 
        Associate statistics from train dataset to valid and test dataset
        Normalize them according to their statistics 
        '''
        # Define train_dataset and normalize it
        print('Tackling Training Set')
        train_dataset = TensorDataset(self.data_train, mini = mini, maxi = maxi, mean=mean, std=std)
        train_dataset = train_dataset.normalize_tensor(self.dims, self.minmaxnorm, self.standardize, reverse = False)

        # Define valid_dataset
        print('Tackling Validation Set')
        valid_dataset = TensorDataset(self.data_valid,mini = train_dataset.mini , maxi = train_dataset.maxi, mean = train_dataset.mean , std = train_dataset.std )

        # Define test_dataset
        print('Tackling Testing Set')
        test_dataset = TensorDataset(self.data_test,mini = train_dataset.mini , maxi = train_dataset.maxi, mean = train_dataset.mean , std = train_dataset.std )
        
        
        # Normalize thank to stats from Training Set 
        valid_dataset = valid_dataset.normalize_tensor(self.dims, self.minmaxnorm, self.standardize, reverse = False)
        test_dataset = test_dataset.normalize_tensor(self.dims, self.minmaxnorm, self.standardize, reverse = False)

        return(train_dataset,valid_dataset,test_dataset)

class InvalidDatesCleaner(object):
    '''
    Object which remove all the forbidden dates / forbidden indices from a array of indices.

    args 
    ----
    invalid_dates :  list of invalid timestamp which should be removed from every dataset
    invalid_indices : list of indices corresponding to the invalid timestamp which should be removed from every dataset 
    df_index : list of time-stamp corresponding to time-slots of the data
    '''
    def __init__(self,invalid_dates : pd.core.indexes.datetimes.DatetimeIndex = None, invalid_indices : np.array = None,df_index : pd.core.indexes.base.Index = None):
        super(InvalidDatesCleaner,self).__init__()
        self.invalid_dates = invalid_dates
        self.invalid_indices = invalid_indices

        if self.invalid_indices is None:
            self.match_date_to_indices(df_index)


    def clean_indices(self,indices):
        mask = np.isin(indices,self.invalid_indices, invert = True)
        return(indices[mask])
    
    def match_date_to_indices(self,df_index):
        self.invalid_indices = df_index.isin(self.invalid_dates).nonzero()[0].tolist()


class FeatureVectorBuilder(object):
    '''
    Shift : 
    1 Week : Mean to predict t+s we need what happened the same day, same hour, same minute, one week ago. 
    which means t+s - 672 

    1 Day : Mean to predict t+s we need what happened the later day, same hour, same minute, 
    which means t+s - 96 

    2 historical length : to predict t+s we need what to know happened the h+s time step earlier, and we can't accès to what happened the s time step earlier
    which means t, t-1, .. t-h   
    '''
    def __init__(self,step_ahead,historical_len,Days,Weeks,Day_nb_steps,Week_nb_steps,shift_from_first_elmt):
        super(FeatureVectorBuilder,self).__init__()

        self.step_ahead = step_ahead
        self.historical_len = historical_len
        self.Days = Days
        self.Weeks = Weeks
        self.Day_nb_steps = Day_nb_steps
        self.Week_nb_steps = Week_nb_steps
        self.shift_from_first_elmt = shift_from_first_elmt
    
    def build_feature_vect(self,tensor: torch.Tensor):
        '''
        args
        ------
        tensor : Torch Tensor of raw data. Shape: [T,N] or [T,N,C,H,W]

        output
        ------ 
        output : [T',N,L] or [T',N,C,H,W,L], with L the total historical length (H+W+D) and T' = T - shift_from_first_elmt 
        '''

        # Shift Values :
        Uwt = [torch.roll(tensor,t*self.Week_nb_steps,0) for t in range(self.Weeks,0,-1)]  
        Udt = [torch.roll(tensor,t*self.Day_nb_steps,0) for t in range(self.Days,0,-1)]
        Ut =  [torch.roll(tensor,self.step_ahead-1+t,0) for t in range(self.historical_len,0,-1)] 

        # Stack 
        U = torch.stack(Uwt+Udt+Ut,dim=-1)

        # Remove the first elements which contains future values
        self.U = U[self.shift_from_first_elmt:]


    def build_target_vect(self, tensor: torch.Tensor):
        '''
        args 
        ------
        step_ahead : number of time-step ahead that we want to predict
        tensor : tensor of raw data, without shift


        output
        ------
        Target Tensor, with shift of step_ahead. 
        '''
        Utarget = torch.unsqueeze(tensor[self.shift_from_first_elmt:],-1) # add last dim
        self.Utarget = Utarget


class DatesVerifFeatureVect(object):
    ''' From dataframe with TimeStamp Index, and historical elements, return df_verif'''
    def __init__(self,df_dates, Weeks = 0, Days = 0, historical_len = 0, step_ahead = 1, time_step_per_hour = 4):

        self.df_dates = df_dates

        self.Weeks = Weeks
        self.Days = Days
        self.historical_len = historical_len
        self.step_ahead = step_ahead

        self.Week_nb_steps = int(7*24*time_step_per_hour)
        self.Day_nb_steps = int(24*time_step_per_hour)
        self.time_step_per_hour = time_step_per_hour

        self.get_shift_from_first_elmt() 
        self.get_df_shifted()

    def get_shift_from_first_elmt(self):
        shift_week = self.Weeks
        shift_day = self.Days
        self.shift_from_first_elmt = int(max(shift_week*24*7*self.time_step_per_hour,
                                shift_day*24*self.time_step_per_hour,
                                self.historical_len+self.step_ahead-1
                                ))

    def shift_dates(self):
        # Weekkly periodic
        Dwt = [self.df_dates.shift((self.Weeks-i)*self.Week_nb_steps) for i in range(self.Weeks)] 
        # Daily periodic
        Ddt = [self.df_dates.shift((self.Days-i)*self.Day_nb_steps) for i in range(self.Days)] 
        # Recent Historic pattern 
        Dt = [self.df_dates.shift(self.step_ahead+(self.historical_len-i)) for i in range(1,self.historical_len+1)] 
        shifted_dates = Dwt+Ddt+Dt
        return(shifted_dates)

    def get_df_shifted(self):
        # Get the shifted "Dates" of Feature Vector and Target
        shifted_dates = self.shift_dates()
        L_shifted_dates = shifted_dates + [self.df_dates]
        Names = [f't-{str(self.Week_nb_steps*(self.Weeks-w))}' for w in range(self.Weeks)] + [f't-{str(self.Day_nb_steps*(self.Days-d))}' for d in range(self.Days)] + [f't-{str(self.historical_len-t)}' for t in range(self.historical_len)]+ [f't+{self.step_ahead-1}']
        self.df_shifted = pd.DataFrame({name:lst['date'] for name,lst in zip(Names,L_shifted_dates)})[self.shift_from_first_elmt:] 

    def identify_forbidden_index(self,invalid_dates):
        '''Get forbidden index of the df_dates and associated forbidden indices of the Torch Tensor'''
        # Mask for dataframe df_verif
        df_shifted_forbiden = pd.concat([self.df_shifted[self.df_shifted[c].isin(invalid_dates)] for c in self.df_shifted.columns])  # Concat forbidden indexes within each columns
        # Identify forbidden df indexes
        self.forbidden_index = df_shifted_forbiden.index.unique()  # drop dupplicates
        # Identify forbidden Tensor Indices 
        self.forbidden_indice_U = self.forbidden_index - self.shift_from_first_elmt  #shift index to get back to corresponding indices

    def get_df_verif(self,invalid_dates):
        # Identify forbidden_index from the df_shifted 
        self.identify_forbidden_index(invalid_dates)
        # Mask shifted from all the forbidden index
        self.df_verif = self.df_shifted.drop(self.forbidden_index)