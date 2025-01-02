import pandas as pd 
import numpy as np
import torch 
import torch.nn as nn 
from constants.paths import USELESS_DATES
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


class TensorLimitsKeeper(object):
    '''
    Object which stores all the limits from cleaned FeatureVector
    '''
    def __init__(self, split_limits,df_dates,df_verif, train_prop,valid_prop, test_prop,step_ahead):
        self.split_limits = split_limits
        self.df_dates = df_dates
        self.df_verif = df_verif
        self.predicted_dates = df_verif.iloc[:,-1]
        self.train_prop = train_prop
        self.valid_prop = valid_prop
        self.test_prop = test_prop
        self.step_ahead = step_ahead

    def get_local_df_verif(self,training_mode):
        '''Set attribute associated to  df_verif_train, df_verif_valid, and df_verif_test '''
        if self.split_limits[f"last_predicted_{training_mode}_date"] is not None:
            
            setattr(self,f"df_verif_{training_mode}",self.df_verif[(self.predicted_dates >= self.split_limits[f"first_predicted_{training_mode}_date"]) & 
                                                                   (self.predicted_dates < self.split_limits[f"last_predicted_{training_mode}_date"]) 
                                                                   ]
            )

    def keep_track_on_df_limits(self,training_mode):
        '''Set attribute to keep track on Train/Valid/Test df limits : first_{training_mode}_date and last_{training_mode}_date  '''

        if getattr(self,f"{training_mode}_prop") > 1e-3:
            setattr(self,f"first_{training_mode}_date",getattr(self,f"df_verif_{training_mode}").iat[0,0])
            setattr(self,f"last_{training_mode}_date",getattr(self,f"df_verif_{training_mode}").iat[-1,-1])

            setattr(self,f"first_predicted_{training_mode}_date",self.split_limits[f"first_predicted_{training_mode}_date"])
            setattr(self,f"last_predicted_{training_mode}_date",self.split_limits[f"last_predicted_{training_mode}_date"])
        else :
            setattr(self,f"first_{training_mode}_date",None)
            setattr(self,f"last_{training_mode}_date",None)

            setattr(self,f"first_predicted_{training_mode}_date",None)
            setattr(self,f"last_predicted_{training_mode}_date",None)

    def get_raw_values_indices(self,training_mode):
        ''' Set attribute to keep trakc on Train/Valid/Test Tensor limits with a list of indices'''
        if getattr(self,f"{training_mode}_prop") > 1e-3:
            reindex = getattr(self, f"df_verif_{training_mode}").stack().unique()
            setattr(self,f"{training_mode}_indices",self.df_dates[self.df_dates['date'].isin(reindex)].index.tolist())

    def get_raw_tensor_input_by_training_mode(self,dataset,training_mode):
        if getattr(self,f"{training_mode}_prop") > 1e-3:
            setattr(dataset,f"{training_mode}_input",dataset.raw_values[getattr(self,f"{training_mode}_indices")])

    def keep_track_on_feature_vect_limits(self,training_mode):
        if getattr(self,f"{training_mode}_prop") > 1e-3:
            attribute_first = self.df_verif.index.get_loc(self.df_verif[self.df_verif[f"t+{self.step_ahead - 1}"] == self.split_limits[f"first_predicted_{training_mode}_date"]].index[0])
            attribute_last = self.df_verif.index.get_loc(self.df_verif[self.df_verif[f"t+{self.step_ahead - 1}"] == self.split_limits[f"last_predicted_{training_mode}_date"]].index[0])
            setattr(self,f"first_{training_mode}_U",attribute_first)
            setattr(self,f"last_{training_mode}_U",attribute_last)
        else: 
            setattr(self,f"first_{training_mode}_U",None)
            setattr(self,f"last_{training_mode}_U",None)


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
    which means t+s - Week_nb_steps     

    1 Day : Mean to predict t+s we need what happened the later day, same hour, same minute, 
    which means t+s - Day_nb_steps 

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
        self.df_dates = df_dates.sort_values('date')
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
        '''
        args:
        -------
        df_shifted: dataframe of date which could be predicted, for each historical (t-w,t-d,t-h,t-h+1,...,t-1) time-step and time-step to predict (t+s-1)
        invalid_dates : list of TimeStamps

        outputs:
        ---------
        forbidden_index : list of index within 'df_shifted' which are related to invalid dates
        forbidden_indice_U: list of indices withing the feature vector 'U' which contains at least one value related to a forbidden dates
        '''

        L_useless_index = []
        for key in USELESS_DATES.keys():
            if key == 'hour':
                useless_object = self.df_shifted.iloc[:,-1].apply(lambda x : x.hour)
                useless_index = useless_object[useless_object.isin(USELESS_DATES[key])].index
            elif key == 'weekday':
                useless_object = self.df_shifted.iloc[:,-1].apply(lambda x : x.weekday())
                useless_index = useless_object[useless_object.isin(USELESS_DATES[key])].index       
            else:
                raise NotImplementedError(f"key {key} has not been implemented as useless index")  
            L_useless_index = L_useless_index+list(useless_index)


        # Mask for dataframe df_verif
        df_shifted_forbiden = pd.concat([self.df_shifted[self.df_shifted[c].isin(invalid_dates)] for c in self.df_shifted.columns])  # Concat forbidden indexes within each columns

        # Identify forbidden df indexes
        self.forbidden_index = df_shifted_forbiden.index.unique() # drop dupplicates

        # Union of Indexes to remove: 
        #self.forbidden_index = list(set(set(L_useless_index)&set(list(df_shifted_forbiden.index.unique()))))  
        self.forbidden_index =  self.forbidden_index.union(pd.Index(L_useless_index)).unique()
        # Identify forbidden Tensor Indices 
        self.forbidden_indice_U = self.forbidden_index - self.shift_from_first_elmt  #shift index to get back to corresponding indices

    def get_df_verif(self,invalid_dates):
        # Identify forbidden_index from the df_shifted 
        self.identify_forbidden_index(invalid_dates)

        # Mask shifted from all the forbidden index
        self.df_verif = self.df_shifted.drop(self.forbidden_index)
        #print("\n>>>>> df_verfi DONE")
        #print('df_verif columns: ',self.df_verif.columns)
        #print(self.df_verif)