import pandas as pd 
import numpy as np
import time
from torch.utils.data import DataLoader
import torch 
import torch.nn as nn 
# Personnal import: 
from metrics import evaluate_metrics
from utilities import get_higher_quantile
try: 
    from ray import tune
except : 
    print('Training and Hyper-parameter tuning with Ray is not possible')


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


class PI_object(object):
    def __init__(self,preds,Y_true,alpha, type_calib = 'CQR',Q = None,T_labels = None):
        super(PI_object,self).__init__()
        self.alpha = alpha
        self.Y_true = Y_true

        if type(Q) == dict:
            Q_tensor = torch.zeros(preds.size(0),preds.size(1),1).to(preds)
            for label in T_labels.unique():
                indices = torch.nonzero(T_labels == label).squeeze()
                try: 
                    Q_tensor[indices,:,0] = Q[label.item()]['Q'][0,:,0]
                except:
                    print(f"No Conformal Calibration value found for {label.item()}. Will be set to 100") 
                    Q_tensor[indices,:,0] = 100
        else : 
            Q_tensor = Q

        self.Q_tensor = Q_tensor
        
        if type_calib == 'CQR':
            self.bands = {'lower':preds[...,0].unsqueeze(-1)-self.Q_tensor, 'upper': preds[...,1].unsqueeze(-1)+self.Q_tensor}
            self.lower = preds[...,0].unsqueeze(-1)-self.Q_tensor
            self.upper = preds[...,1].unsqueeze(-1)+self.Q_tensor

        if type_calib =='classic':
            self.bands = {'lower':preds[...,0].unsqueeze(-1), 'upper': preds[...,1].unsqueeze(-1)}
            self.lower = preds[...,0].unsqueeze(-1)
            self.upper = preds[...,1].unsqueeze(-1)

        self.MPIW()
        self.PICP()
    
    
    def MPIW(self):
        self.mpiw = torch.mean(self.bands['upper']-self.bands['lower']).item()
        return(self.mpiw)
    def PICP(self):
        self.picp = torch.sum((self.lower<self.Y_true)&(self.Y_true<self.upper)).item()/torch.prod(torch.Tensor([s for s in self.lower.size()])).item()
        return(self.picp)

        
        

class DictDataLoader(object):
    ## DataLoader Classique pour le moment, puis on verra pour faire de la blocked cross validation
    '''
    args
    -----
    validation -> classic (basic one, with train_prop first percent of the dataset for training) 
                  blocked (for Blocked K-fold Cross validation)
    '''
    def __init__(self,U,Utarget,train_prop,valid_prop,validation = 'classic', shuffle = True, calib_prop = None,time_slots = None):
        super().__init__()
        self.validation = validation
        self.train_prop = train_prop
        self.valid_prop = valid_prop
        self.U = U
        self.Utarget = Utarget 
        self.dataloader = {}
        self.shuffle = shuffle
        self.calib_prop = calib_prop
        self.time_slots = time_slots

    def train_test_split(self,U,Utarget,time_slots,validation = 'classic'):
        n = U.size(0)

        if validation == 'sliding_window':
            train_prop,valid_prop  = self.train_prop*1/(self.train_prop+self.valid_prop),self.valid_prop*1/(self.train_prop+self.valid_prop)
        if validation == 'classic':
            train_prop,valid_prop  = self.train_prop,self.valid_prop     

        train_idx,valid_idx = int(n*train_prop),int(n*(train_prop+valid_prop))

        # Train, Valid, Test set
        train_set,valid_set,test_set = U[:train_idx],U[train_idx:valid_idx],U[valid_idx:]
        # Target set
        train_target, valid_target, test_target = Utarget[:train_idx],Utarget[train_idx:valid_idx],Utarget[valid_idx:]
        # Time_slots set
        time_slots_train,time_slots_valid,time_slots_test = time_slots[:train_idx], time_slots[train_idx:valid_idx], time_slots[valid_idx:]
        return(train_set,valid_set,test_set,train_target, valid_target, test_target,time_slots_train,time_slots_valid,time_slots_test)

    def fill_data_loader_dict(self,set_of_dataset,batch_size, only_test = False):
        train_set,valid_set,test_set,train_target, valid_target, test_target,time_slots_train,time_slots_valid,time_slots_test = set_of_dataset

        if self.calib_prop is None: 
            if only_test :
                Sequences,Targets,Time_slots_list,Names = [test_set],[test_target],[time_slots_test],['test']
            else:
                Sequences,Targets,Time_slots_list,Names = [train_set,valid_set,test_set],[train_target,valid_target,test_target],[time_slots_train,time_slots_valid,time_slots_test],['train','validate','test']
            for dataset,target,L_time_slot,training_mode in zip(Sequences,Targets,Time_slots_list,Names):
                    self.dataloader[training_mode] = DataLoader(list(zip(dataset,target,L_time_slot)),batch_size=batch_size, shuffle = (True if ((training_mode == 'train') & self.shuffle ) else False)) if len(dataset) > 0 else None
        else : 
            indices = torch.randperm(train_set.size(0))
            split = int(train_set.size(0)*self.calib_prop)
            proper_set_x,proper_set_y = train_set[indices[:split]],train_target[indices[:split]]
            calib_set_x,calib_set_y = train_set[indices[split:]],train_target[indices[split:]]
            time_slots_proper,time_slots_calib = time_slots_train[indices[:split]], time_slots_train[indices[split:]]

            if only_test :
                Sequences,Targets,Time_slots_list,Names = [test_set],[test_target],[time_slots_test],['test']
            else:
                Sequences,Targets,Time_slots_list,Names = [proper_set_x,valid_set,test_set,calib_set_x],[proper_set_y,valid_target,test_target,calib_set_y],[time_slots_proper,time_slots_valid,time_slots_test,time_slots_calib],['train','validate','test','cal']

            for dataset,target,L_time_slot,training_mode in zip(Sequences,Targets,Time_slots_list,Names):
                self.dataloader[training_mode] = DataLoader(list(zip(dataset,target,L_time_slot)),batch_size=(dataset.size(0) if training_mode=='cal' else batch_size), shuffle = (True if ((training_mode == 'train') & self.shuffle ) else False)) if len(dataset) > 0 else None   

    def get_one_dataloader(self,U,Utarget,time_slots,batch_size:int, k = None, K_fold = None, validation = 'classic'):
        # Il faut gérer les train_prop / valid_prop / calib_prop
        # Dans le cas de la sliding window, il me faut train_prop + valid_prop = 1  car on ne veut pas de test.


        set_of_dataset = self.train_test_split(U,Utarget,time_slots,validation)

        if validation == 'sliding_window': 
            n = U.size(0)
            U_sliced,Utarget_sliced,time_slots_sliced = U[int((k/K_fold)*n):int(((k+1)/K_fold)*n)],Utarget[int((k/K_fold)*n):int(((k+1)/K_fold)*n)],time_slots[int((k/K_fold)*n):int(((k+1)/K_fold)*n)]
            set_of_sliced_dataset= self.train_test_split(U_sliced,Utarget_sliced,time_slots_sliced,validation)

            # Add full Test DataLoader and a sliced DataLoader
            self.fill_data_loader_dict(set_of_sliced_dataset,batch_size, only_test = False)
            self.fill_data_loader_dict(set_of_dataset,batch_size, only_test = True)      
                
        elif validation == 'classic':
            self.fill_data_loader_dict(set_of_dataset,batch_size,only_test = False)

        return(self.dataloader)
    
    def get_dictdataloader(self,batch_size:int, K_fold = None):
        if self.validation == 'classic':
            return(self.get_one_dataloader(self.U,self.Utarget,self.time_slots,batch_size))

        if self.validation == 'sliding_window': # blocked k fold cross validation
            return([self.get_one_dataloader(self.U,self.Utarget,self.time_slots,batch_size,k = k, K_fold= K_fold,validation = self.validation) for k in range(K_fold)])

class MultiModelTrainer(object):
    def __init__(self,model_list,dataloader_list,args,optimizer_list,loss_function,scheduler,args_embedding,ray=False,alpha = None):
        super(MultiModelTrainer).__init__()
        self.Trainers = [Trainer(model,dataloader,args,optimizer,loss_function,scheduler,ray,args_embedding) for dataloader,model,optimizer in zip(dataloader_list,model_list,optimizer_list)]
        self.Loss_train =  torch.Tensor().to(args.device) #{k:[] for k in range(len(dataloader_list))}
        self.Loss_valid = torch.Tensor().to(args.device) #{k:[] for k in range(len(dataloader_list))}    
        self.alpha = alpha 
        self.picp = []   
        self.mpiw = []

    def K_fold_validation(self):
        for k,trainer in enumerate(self.Trainers):
            # Train valid model 
            print(f"K_fold {k}")
            trainer.train_and_valid(mod = 10000)

            # Add Loss 
            self.Loss_train = torch.cat([self.Loss_train,torch.Tensor(trainer.train_loss).unsqueeze(0)],axis =  0) 
            self.Loss_valid = torch.cat([self.Loss_valid,torch.Tensor(trainer.valid_loss).unsqueeze(0)],axis =  0) 
            # Testing
            if self.alpha is not None:
                preds,Y_true,_ = trainer.test_prediction(training_mode = 'test')
                pi = PI_object(preds,Y_true,self.alpha,type_calib = 'classic')
                self.picp.append(pi.picp)
                self.mpiw.append(pi.mpiw)

            print(f"Last Train Loss: {trainer.train_loss[-1]},Last Valid Loss: {trainer.valid_loss[-1]},PICP: {pi.picp}, MPIW: {pi.mpiw}")
        mean_picp = torch.Tensor(self.picp).mean()
        mean_mpiw = torch.Tensor(self.mpiw).mean() 
        assert len(self.Loss_train.mean(dim = 0)) == len(trainer.train_loss), 'Mean on the wrong axis'
        mean_last_train_loss = self.Loss_train.mean(dim = 0)[-1]  #.item()
        mean_last_valid_loss = self.Loss_valid.mean(dim = 0)[-1]  #.item()

        min_mean_train_loss = self.Loss_train.mean(dim = 0).min()  #.item()
        min_mean_valid_loss = self.Loss_valid.mean(dim = 0).min()  #.item()
        return(mean_picp,mean_mpiw,mean_last_train_loss,mean_last_valid_loss,min_mean_train_loss,min_mean_valid_loss)       


    

class Trainer(object):
        ## Trainer Classique pour le moment, puis on verra pour faire des Early Stop 
    def __init__(self,model,dataloader,args,optimizer,loss_function,scheduler = None, ray = False,args_embedding  =None, save_path = None):
        super().__init__()
        self.dataloader = dataloader
        self.training_mode = 'train'
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.model = model 
        self.scheduler = scheduler
        self.train_loss = []
        self.valid_loss = []
        self.calib_loss =[]
        self.epochs = args.epochs
        self.device = args.device
        self.quantile_method = args.quantile_method 
        self.conformity_scores_type = args.conformity_scores_type
        self.ray = ray
        self.args_embedding = args_embedding
        self.save_path = save_path 
        self.best_valid = np.inf

    def save_best_model(self,checkpoint,epoch):
        ''' Save best model in .pkl format'''
        checkpoint.update(epoch=epoch, state_dict=self.model.state_dict())
        torch.save(checkpoint, f"{self.save_path}")    

    def train_and_valid(self,mod = 10, alpha = None,dataset = None):
        print(f'start training')
        checkpoint = {'epoch':0, 'state_dict':self.model.state_dict()}
        for epoch in range(self.epochs):
            t0 = time.time()
            # Train and Valid each epoch 
            self.training_mode = 'train'
            self.model.train()   #Activate Dropout 
            self.loop()
            self.training_mode = 'validate'
            self.model.eval()   # Desactivate Dropout 
            self.loop()


            if (self.valid_loss[-1] < self.best_valid) & (self.save_path is not None):
                self.best_valid = self.valid_loss[-1]
                self.save_best_model(checkpoint,epoch)

            # Keep track on Metrics
            if self.ray : 
                # Calibration 
                Q = self.conformal_calibration(alpha,dataset,conformity_scores_type = self.conformity_scores_type,quantile_method = self.quantile_method)  
                # Testing
                preds,Y_true,T_labels = self.test_prediction(training_mode = 'validate')
                # get PI
                pi = self.CQR_PI(preds,Y_true,alpha,Q,T_labels)
                # Report usefull metrics
                tune.report(Loss_model = self.valid_loss[-1], MPIW = pi.mpiw, PICP = pi.picp) 

            # Update scheduler after each Epoch 
            if self.scheduler is not None:
               self.scheduler.step()

            if epoch%mod==0:
                print(f"epoch: {epoch} \n min\epoch : {'{0:.2f}'.format((time.time()-t0)/60)}")
            if epoch == 1:
                print(f"Estimated time for training: {'{0:.1f}'.format(self.epochs*(time.time()-t0)/60)}min ")


    def loop(self,):
        loss_epoch,nb_samples = 0,0
        with torch.set_grad_enabled(self.training_mode=='train'):
            for x_b,y_b,t_b in self.dataloader[self.training_mode]:
                x_b,y_b,t_b = x_b.to(self.device),y_b.to(self.device),t_b.to(self.device)
                #Forward 
                if self.args_embedding is not None: 
                    pred = self.model(x_b,t_b.long())
                else:
                    pred = self.model(x_b)
                loss = self.loss_function(pred,y_b)

                # Back propagation (after each mini-batch)
                if self.training_mode == 'train': 
                    loss = self.backpropagation(loss)

                # Keep track on metrics 
                nb_samples += x_b.shape[0]
                loss_epoch += loss.item()*x_b.shape[0]
        self.update_loss_list(loss_epoch,nb_samples,self.training_mode)

    def conformal_calibration(self,alpha,dataset,conformity_scores_type = 'max_residual',quantile_method = 'classic',week_group = None, hour_group = None):
        ''' 
        Quantile estimator (i.e NN model) is trained on the proper set
        Conformity scores computed with quantile estimator on the calibration set
        And then the empirical th-quantile Q is computed with the conformity scores and quantile function

        inputs
        -------
        - alpha : is the miscoverage rate. such as  P(Y in C(X)) >= 1- alpha 
        - dataset : DataSet object. Allow us to unormalize tensor
        '''
        self.model.eval()
        with torch.no_grad():
            data = [[x_b,y_b,t_b] for  x_b,y_b,t_b in self.dataloader['cal']]
            X_cal,Y_cal,T_cal = torch.cat([x_b for [x_b,_,_] in data]).to(self.device),torch.cat([y_b for [_,y_b,_] in data]).to(self.device),torch.cat([t_b for [_,_,t_b] in data]).to(self.device)

            #Forward 
            if self.args_embedding is not None: 
                preds = self.model(X_cal,T_cal.long())
            else:
                preds = self.model(X_cal) 

            if len(preds.size()) == 2:
                preds = preds.unsqueeze(1)


            # get lower and upper band
            if preds.size(-1) == 2:
                lower_q,upper_q = preds[...,0].unsqueeze(-1),preds[...,1].unsqueeze(-1)   # The Model return ^q_l and ^q_u associated to x_b
        
            elif preds.size(-1) == 1:
                lower_q,upper_q = preds,preds 
            else:
                raise ValueError(f"Shape of model's prediction: {preds.size()}. Last dimension should be 1 or 2.")
            
            # unormalized lower and upper band 
            lower_q, upper_q = dataset.unormalize_tensor(lower_q,device = self.device),dataset.unormalize_tensor(upper_q,device = self.device)
            Y_cal = dataset.unormalize_tensor(Y_cal,device = self.device)

            # Confority scores and quantiles
            if conformity_scores_type == 'max_residual':
                self.conformity_scores = torch.max(lower_q-Y_cal,Y_cal-upper_q).to(self.device) # Element-wise maximum        #'max(lower_q-y_b,y_b-upper_q)' is the quantile regression error function
            if conformity_scores_type == 'max_residual_plus_middle':
                print("|!| Conformity scores computation is not based on 'max(ql-y, y-qu)'")
                self.conformity_scores = torch.max(lower_q-Y_cal,Y_cal-upper_q) + ((lower_q>Y_cal)(upper_q<Y_cal))*(upper_q - lower_q)/2  # Element-wise maximum        #'max(lower_q-y_b,y_b-upper_q)' is the quantile regression error function



            # Get Quantile :
            if quantile_method == 'classic':  
                quantile_order = torch.Tensor([np.ceil((1 - alpha)*(X_cal.size(0)+1))/X_cal.size(0)]).to(self.device)
                #Q = torch.quantile(self.conformity_scores, quantile_order, dim = 0).to(self.device) #interpolation = 'higher'
                Q = get_higher_quantile(self.conformity_scores,quantile_order,device = self.device)
                output = Q
            if quantile_method == 'weekday_hour':
                calendar_class = torch.cat([t_b for [_,_,t_b] in data])
                dic_label2Q = {}


                # Compute quantile for each calendar class : 
                nb_label_with_quantile_1 = 0
                for label in calendar_class.unique():
                    indices = torch.nonzero(calendar_class == label,as_tuple = True)[0]
                    quantile_order = torch.Tensor([np.ceil((1 - alpha)*(indices.size(0)+1))/indices.size(0)]).to(self.device)  # Quantile for each class, so the quantile order is different as each class has a different length
                    quantile_order = min(torch.Tensor([1]).to(self.device),quantile_order)
                    if quantile_order == 1: 
                        nb_label_with_quantile_1 +=1
                        #print(f"label {label} has only {indices.size(0)} elements in his class. We then use quantile order = 1")
                    conformity_scores_i = self.conformity_scores[indices]
                    scores_counts = conformity_scores_i.size(0)
                    Q_i = get_higher_quantile(conformity_scores_i,quantile_order,device = self.device)
                    #Q_i = torch.quantile(conformity_scores_i, quantile_order, dim = 0)#interpolation = 'higher'
                    dic_label2Q[label.item()]= {'Q': Q_i,'count':scores_counts}
                print(f"Proportion of label with quantile order set to 1: {'{:.1%}'.format(nb_label_with_quantile_1/len(calendar_class.unique()))}")
                output = dic_label2Q

        return(output)
    
    def CQR_PI(self,preds,Y_true,alpha,Q,T_labels = None):
        pi = PI_object(preds,Y_true,alpha,type_calib = 'CQR',Q=Q,T_labels = T_labels,device = self.device)
        self.pi = pi
        return(pi)


    def backpropagation(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return(loss)
    
    def test_prediction(self,allow_dropout = False,training_mode = 'test'):
        self.training_mode = training_mode
        if allow_dropout:
            self.model.train()
        else: 
            self.model.eval()
        with torch.no_grad():
            # Au lieu de          Pred = torch.cat([self.model(x_b.to(self.device)) for x_b,y_b in self.dataloader[self.training_mode]]) // Y_true = torch.cat([y_b.to(self.device) for x_b,y_b in self.dataloader[self.training_mode]])
            data = [[x_b,y_b,t_b] for  x_b,y_b,t_b in self.dataloader[training_mode]]
            X,Y_true,T_labels= torch.cat([x_b for [x_b,_,_] in data]).to(self.device),torch.cat([y_b for [_,y_b,_] in data]).to(self.device), torch.cat([t_b for [_,_,t_b] in data]).to(self.device)
            
            if self.args_embedding is not None: 
                Pred = self.model(X,T_labels.long())
            else:
                Pred = self.model(X) 
                
        return(Pred,Y_true,T_labels)

    def testing(self,dataset,metrics= ['mse','mae'], allow_dropout = False):
        (test_pred,Y_true,T_labels) = self.test_prediction(allow_dropout)  # Get Normalized Pred and Y_true

        test_pred = dataset.unormalize_tensor(test_pred, device = self.device)
        Y_true = dataset.unormalize_tensor(Y_true, device = self.device)

        df_metrics = evaluate_metrics(test_pred,Y_true,metrics)

        return(test_pred,Y_true,T_labels,df_metrics)  
    
    def update_loss_list(self,loss_epoch,nb_samples,training_mode):
        if training_mode == 'train':
            self.train_loss.append(loss_epoch/nb_samples)
        elif training_mode == 'validate':
            self.valid_loss.append(loss_epoch/nb_samples)
        elif training_mode == 'calibrate':
            self.calib_loss.append(loss_epoch/nb_samples)


class DataSet(object):
    '''
    attributes
    -------------
    df : contain the current df you are working on. It's the full df, normalized or not
    init_df : contain the initial df, no normalized. It's the full initial dataset.
    cleaned_df : contain the init_df, without the invalid_dates
    df_train : contain the train_prop fist % of the cleaned_df 
    '''
    def __init__(self,df,init_df = None,mini= None, maxi = None, mean = None, normalized = False,time_step_per_hour = None,df_train = None,cleaned_df = None):
        self.length = len(df)
        self.df = df
        self.columns = df.columns
        self.normalized = normalized
        self.time_step_per_hour = time_step_per_hour
        self.df_dates = pd.DataFrame(self.df.index,index = np.arange(len(self.df)),columns = ['date'])
        self.df_train = df_train
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
        self.cleaned_df = cleaned_df
        
    def bijection_name_indx(self):
        colname2indx = {c:k for k,c in enumerate(self.columns)}
        indx2colname = {k:c for k,c in enumerate(self.columns)}
        return(colname2indx,indx2colname)
    
    def minmaxnorm(self,x,reverse = False):
        if reverse:
            x = x*(self.maxi - self.mini) +self.mini
        else :
            x = (x-self.mini)/(self.maxi-self.mini)
        return x 
    
    def remove_invalid_dates(self,tmps_df,invalid_dates):
        if invalid_dates is not None:
            invalid_dates = invalid_dates.intersection(tmps_df.index)
            tmps_df = tmps_df.drop(invalid_dates)
        self.cleaned_df = tmps_df
        return(self.cleaned_df)
    
    def minmax_normalize_df(self,tmps_df):
        self.mini = tmps_df.min()
        self.maxi = tmps_df.max()
        self.mean = tmps_df.mean()

        # Keep track on data used for training : 
        self.df_train = tmps_df

        # Normalize : 
        normalized_df = self.minmaxnorm(self.df)  # Normalize the entiere dataset

        # Update state : 
        self.df = normalized_df
        self.normalized = True

    def normalize_df(self, train_prop,minmaxnorm = True):
        assert self.normalized == False, 'Dataframe might be already normalized'
        tmps_df = self.cleaned_df if self.cleaned_df is not None else self.init_df
        tmps_df = tmps_df[:int(train_prop*self.length)]  # Slicing to compute min max on training df

        if minmaxnorm:
            self.minmax_normalize_df(tmps_df)
        else:
            raise Exception('Normalization has not been coded')
        
    def unormalize_df(self,minmaxnorm):
        assert self.normalized == True, 'Dataframe might be already UN-normalized'
        if minmaxnorm:
            self.df = self.minmaxnorm(self.df,reverse = True)
        self.normalized = False


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
    
    def split_K_fold(self,K_fold,train_prop,valid_prop,normalized):
        '''
        Split la DataSet Initiale en K-fold

        Problème : 
        Lors qu'on utilise la methode 
            'get_feature_vect'   (pour obtenir les sequences sur toute la DataFrame)
        Il va y avoir un shift de 1 semaine (si w = 1), et donc tout va se passer comme si la première semaine de chaque fold était à chaque fois impossible à prédire.
        - - - Maintenant, si on ajoute cette semaine là dans le dataset, cela veut dire qu'il faut la compter dans les données pour la Normaliser - - - 
        En fait, avec la sliding-windows, on perd % = = = = K-fold * Semaine de prédiction= = = = %

        1- Lors de la boucle sur les K-fold, il faudrait non pas prendre df[int((k/K_fold)*n):int(((k+1)/K_fold)*n)] 
           Mais plutôt df[int((k/K_fold)*n)-1 semaine:int(((k+1)/K_fold)*n)]  (- le décalage max).

        2 - On devra alors utiliser dataset.normalize_df sur chacun des dataset
            en précisant une train_prop = train_prop * 1/(train_prop + Valid_prop), puisqu'on ne voudra pas garder de test_set à côté. (il est déjà conservé sur Novembre/Decembre)


        3 - 'get_feature_vect' et les autres sont ensuite bon.  
            On va ensuite passer dans 'get_invalid_indx'  puis dans  'remove_indices' pour supprimer les séquences qui nous dérange.

        4 - Puis on fera des DataLoader + .get_dict_data_loader()


        '''
        Datasets = []
        # Récupère la df (On garde les valeurs interdite pour le moment, on les virera après. Il est important de les virer pour la normalisation, pour pas Normaliser la donnée avec des valeurs qui n'ont pas de sens.)
        df = self.df

        # Fait la 'Hold-Out' séparation, pour enlever les dernier mois de TesT
        df = df[:int((train_prop+valid_prop)*len(df))]  

        # Récupère la Taille de cette DataFrame
        n = len(df)

        # Découpe la dataframe en K_fold 
        for k in range(K_fold):
            # Slicing 
            df_tmps = df[int((k/K_fold)*n):int(((k+1)/K_fold)*n)]
            # Récupération d'une dataset associée 
            dataset_tmps = DataSet(df_tmps,init_df = None,mini= None, maxi = None, mean = None, normalized = normalized,time_step_per_hour = self.time_step_per_hour,df_train = None)
            dataset_tmps.normalize_df(train_prop*1/(train_prop+valid_prop),minmaxnorm = True)
            # Ajoute l'objet Dataset-k à la liste 
            Datasets.append(dataset_tmps)

        return(Datasets)



    def unormalize(self,timeserie):
        if not(self.normalized):
            print('The df might be already unormalized')
        return(timeserie*(self.maxi - self.mini)+self.mini)
    
    def unormalize_tensor(self,tensor, axis = -1,device = 'cpu'):
        maxi_ = torch.Tensor(self.maxi.values).unsqueeze(axis).to(device)
        mini_ = torch.Tensor(self.mini.values).unsqueeze(axis).to(device)
        unormalized = tensor*(maxi_ - mini_)+mini_
        return unormalized
    
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
             
            self.shift_from_first_elmt = int(max(Weeks*24*7*self.time_step_per_hour,
                                    Days*24*self.time_step_per_hour,
                                    historical_len+step_ahead-1
                                    ))
            
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