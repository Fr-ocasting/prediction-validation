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
        train_set,valid_set,test_set = U[:train_idx],U[train_idx:valid_idx],U[valid_idx:]
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
        
    def bijection_name_indx(self):
        colname2indx = {c:k for k,c in enumerate(self.columns)}
        indx2colname = {k:c for k,c in enumerate(self.columns)}
        return(colname2indx,indx2colname)
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
