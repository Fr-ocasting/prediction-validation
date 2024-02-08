from torch.utils.data import DataLoader
import torch 
import torch.nn as nn 
import pandas as pd 
import numpy as np

class DictDataLoader(object):
    ## DataLoader Classique pour le moment, puis on verra pour faire de la blocked cross validation
    def __init__(self,U,Utarget,train_prop,valid_prop,validation = 'classic', shuffle = True):
        super().__init__()
        self.validation = validation
        self.train_prop = train_prop
        self.valid_prop = valid_prop
        self.U = U
        self.Utarget = Utarget 
        self.dataloader = {}
        self.shuffle = shuffle

    def train_test_split(self):
        n = self.U.shape[0]
        train_idx,valid_idx = int(n*self.train_prop),int(n*(self.train_prop+self.valid_prop))
        train_set,valid_set,test_set = self.U[:train_idx],self.U[train_idx:valid_idx],self.U[valid_idx:]
        train_target, valid_target, test_target = self.Utarget[:train_idx],self.Utarget[train_idx:valid_idx],self.Utarget[valid_idx:]
        return(train_set,valid_set,test_set,train_target, valid_target, test_target)

    def get_dictdataloader(self,batch_size:int):
        train_set,valid_set,test_set,train_target, valid_target, test_target= self.train_test_split()
        for dataset,target,training_mode in zip([train_set,valid_set,test_set],[train_target,valid_target,test_target],['train','validate','test']):
            self.dataloader[training_mode] = DataLoader(list(zip(dataset,target)),batch_size=batch_size, shuffle = (True if ((training_mode == 'train') & self.shuffle ) else False))
        return(self.dataloader)
    

class Trainer(object):
        ## Trainer Classique pour le moment, puis on verra pour faire des Early Stop 
    def __init__(self,model,dataloader,epochs,optimizer,loss_function,scheduler = None):
        super().__init__()
        self.dataloader = dataloader
        self.training_mode = 'train'
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.model = model 
        self.scheduler = scheduler
        self.train_loss = []
        self.valid_loss = []
        self.epochs = epochs

    def train_and_valid(self,):
        for epoch in range(self.epochs):
            # Train and Valid each epoch 
            self.training_mode = 'train'
            self.model.train()   #Activate Dropout 
            self.loop()
            self.training_mode = 'validate'
            self.model.eval()   # Desactivate Dropout 
            self.loop()

            # Update scheduler after each Epoch 
            if self.scheduler is not None:
               self.scheduler.step()

    def loop(self,):
        loss_epoch,nb_samples = 0,0
        with torch.set_grad_enabled(self.training_mode=='train'):
            for x_b,y_b in self.dataloader[self.training_mode]:
                #Forward 
                pred = self.model(x_b)
                loss = self.loss_function(pred,y_b)

                # Back propagation (after each mini-batch)
                if self.training_mode == 'train': 
                    loss = self.backpropagation(loss)

                # Keep track on metrics 
                nb_samples += x_b.shape[0]
                loss_epoch += loss.item()*nb_samples
        self.update_loss_list(loss_epoch,nb_samples,self.training_mode)


    def backpropagation(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return(loss)
    
    def test(self):
        self.training_mode = 'test'
        self.model.eval()
        with torch.no_grad():
            Pred = torch.cat([self.model(x_b) for x_b,y_b in self.dataloader[self.training_mode]])
            Y_true = torch.cat([y_b for x_b,y_b in self.dataloader[self.training_mode]])
        return(Pred,Y_true)
    
    def update_loss_list(self,loss_epoch,nb_samples,training_mode):
        if training_mode == 'train':
            self.train_loss.append(loss_epoch/nb_samples)
        elif training_mode == 'validate':
            self.valid_loss.append(loss_epoch/nb_samples)


class DataSet(object):
    def __init__(self,df,init_df = None,mini= None, maxi = None, mean = None, normalized = False,time_step_per_hour = None):
        self.length = len(df)
        self.df = df
        self.columns = df.columns
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

        self.shift_from_first_elmt = None
        self.U = None
        self.Utarget = None
        self.df_verif = None
        self.invalid_indx_df = None
        self.remaining_dates = None
        self.step_ahead = None
        self.Weeks = None
        self.Days = None
        self.historical_len = None
        
    def bijection_name_indx(self):
        colname2indx = {c:k for k,c in enumerate(self.columns)}
        indx2colname = {k:c for k,c in enumerate(self.columns)}
        return(colname2indx,indx2colname)
    
    def minmaxnorm(self,x):
        return(x-self.mini)/(self.maxi-self.mini)
    
    def normalize_df(self, train_prop, invalid_dates = None, minmaxnorm = True):
        if self.normalized:
            print('The df might be already normalized')
        else:
            if minmaxnorm:
                tmps_df = self.init_df[:int(train_prop*self.length)]
                if invalid_dates is not None:
                    tmps_df = tmps_df.drop(invalid_dates)
                self.mini = tmps_df.min()
                self.maxi = tmps_df.max()
                self.mean = tmps_df.mean()
                normalized_df = minmaxnorm(self.init_df)
            normalized_dataset = DataSet(normalized_df,init_df = self.init_df,mini = self.mini, maxi = self.maxi, mean = self.mean,normalized=True,time_step_per_hour=self.time_step_per_hour)
            return(normalized_dataset)
    
    def normalize_tensor(self, minmaxnorm = True):
        if self.normalized:
            print('The df might be already normalized')

    def remove_indices(self,invalid_indices_tensor):
        ''' Remove the invalid sequences matching to invalid dates.
        Sequences have already been shifted, so there is (len(df_init)-shift_from_first_elmt elements) elements left.
        - The first index of df_verif is 0+shift_from_first_elmt
        - The last index of df_verif is len(df_init)
        We then have to remove the invalid sequences from U, and keep the remaining dates from 'df_verif'
        '''
        selected_indices = [e for e in np.arange(self.U.shape[0]) if e not in invalid_indices_tensor]  
        selected_dates_index = [e for e in np.arange(self.shift_from_first_elmt,self.U.shape[0]+self.shift_from_first_elmt) if e not in self.invalid_indx_df]
        self.U = self.U[selected_indices]
        self.Utarget = self.Utarget[selected_indices]
        self.remaining_dates = self.df_verif.loc[selected_dates_index,[f't+{self.step_ahead-1}']]
        return(self.U,self.Utarget,self.remaining_dates)

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
    
    def get_invalid_indx(self,invalid_dates:list,df_verif:pd.DataFrame):
        '''invalid_dates:  list of Tmestamp dates 
        from a list of dates, return 'invalid_indices_tensor', which correspond to the forbidden indices in the Tensor
        and return 'invalid_indx_df' which correspond to the forbidden index within the dataframe (where the first index can be > 0)'''
        # Get all the row were the invalid dates are used 
        df_verif_forbiden = pd.concat([df_verif[df_verif[c].isin(invalid_dates)] for c in df_verif.columns])

        # Get the associated index 
        self.invalid_indx_df = df_verif_forbiden.index

        # Shift them in relation to the tensor 
        invalid_indices_tensor = self.invalid_indx_df - self.shift_from_first_elmt

        return(invalid_indices_tensor,self.invalid_indx_df)


    def get_feature_vect(self,step_ahead,historical_len,Days,Weeks):
        if self.time_step_per_hour is None :
            raise Exception('Number of time steps per hour as not been defined. Please use FeatureVector.time_step_per_hour ')
        
        else : 
             # Update Dataset attributes
            self.step_ahead = step_ahead
            self.historical_len = historical_len
            self.Days = Days
            self.Weeks = Weeks
             
            self.shift_from_first_elmt = max(Weeks*24*7*self.time_step_per_hour,
                                    Days*24*self.time_step_per_hour,
                                    historical_len+step_ahead-1
                                    )
            
            # Get the shifted "Dates" of Feature Vector and Target
            (shifted_values,shifted_dates) = self.shift_data(step_ahead,historical_len,Weeks,Days)
            L_shifted_dates = shifted_dates + [self.df_dates]
            Names = [f't-{str(self.Week_nb_steps*(Weeks-w))}' for w in range(Weeks)] + [f't-{str(self.Day_nb_steps*(Days-d))}' for d in range(Days)] + [f't-{str(historical_len-t)}' for t in range(historical_len)]+ [f't+{step_ahead-1}']
            df_verif = pd.DataFrame({name:lst['date'] for name,lst in zip(Names,L_shifted_dates)})[self.shift_from_first_elmt:]

            # Get Feature Vector and Target 
            U = torch.cat(shifted_values,dim=2)[:][self.shift_from_first_elmt:]
            Utarget = torch.unsqueeze(torch.Tensor(self.df.values),2)[self.shift_from_first_elmt:]

            # Update Dataset attributes
            self.U = U
            self.Utarget = Utarget
            self.df_verif = df_verif
            self.remaining_dates = df_verif[[f't+{step_ahead-1}']]

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
