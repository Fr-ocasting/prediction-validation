import pandas as pd 
import numpy as np
import time
from torch.utils.data import Dataset,DataLoader
import torch 
import torch.nn as nn 
import os 
import pkg_resources


from torch.cuda.amp import autocast,GradScaler

# Personnal import: 
from chrono import Chronometer
from profiler import print_memory_usage,get_cpu_usage
from metrics import evaluate_metrics
from utilities import get_higher_quantile
from datetime import timedelta
from split_df import train_valid_test_split_iterative_method
from calendar_class import get_time_slots_labels
from PI_object import PI_object
from paths import save_folder
try :
    from plotting_bokeh import generate_bokeh
except:
    print('no plotting bokeh available')
from save_results import update_results_df, results2dict,Dataset_get_save_folder,read_object,save_object,save_best_model_and_update_json,load_json_file,get_trial_id
try: 
    from ray import tune,train
    ray_version = pkg_resources.get_distribution("ray").version
    if ray_version.startswith('2.7'):
        report = train.report
    else:
        report = tune.report
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

class CustomDataset(Dataset):
    def __init__(self,X,Y,*T):
        self.X = X
        self.Y = Y
        self.T = T
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        T_data = [t[idx] for t in self.T]
        return self.X[idx], self.Y[idx], *T_data

class DictDataLoader(object):
    ## DataLoader Classique pour le moment, puis on verra pour faire de la blocked cross validation
    '''
    args
    -----
    '''
    def __init__(self,dataset,args):
        super().__init__()
        self.dataloader = {}
        self.calib_prop = args.calib_prop
        self.dataset = dataset
        self.args = args

    def get_dictdataloader(self,batch_size):
        if self.calib_prop is None: 
            Sequences = [self.dataset.U_train,self.dataset.U_valid,self.dataset.U_test]
            Targets = [self.dataset.Utarget_train,self.dataset.Utarget_valid,self.dataset.Utarget_test]
            Time_slots_list = [self.dataset.time_slots_train,self.dataset.time_slots_valid,self.dataset.time_slots_test]
            Names = ['train','validate','test']

        else : 
            indices = torch.randperm(self.dataset.U_train.size(0)) 
            split = int(self.dataset.U_train.size(0)*self.calib_prop)

            self.dataset.indices_cal = indices[split:]
            self.dataset.indices_train = indices[:split]

            proper_set_x,proper_set_y = self.dataset.U_train[self.dataset.indices_train],self.dataset.Utarget_train[self.dataset.indices_train]
            calib_set_x,calib_set_y = self.dataset.U_train[self.dataset.indices_cal],self.dataset.Utarget_train[self.dataset.indices_cal]
            time_slots_proper = {calendar_class: self.dataset.time_slots_train[calendar_class][self.dataset.indices_train] for calendar_class in range(len(self.dataset.nb_class)) } 
            time_slots_calib = {calendar_class: self.dataset.time_slots_train[calendar_class][self.dataset.indices_cal] for calendar_class in range(len(self.dataset.nb_class))}


            Sequences = [proper_set_x,self.dataset.U_valid,self.dataset.U_test,calib_set_x]
            Targets = [proper_set_y,self.dataset.Utarget_valid,self.dataset.Utarget_test,calib_set_y]
            Time_slots_list = [time_slots_proper,self.dataset.time_slots_valid,self.dataset.time_slots_test,time_slots_calib]
            Names = ['train','validate','test','cal']
            
        for feature_vector,target,L_time_slot,training_mode in zip(Sequences,Targets,Time_slots_list,Names):
            if feature_vector is not None:
                inputs = CustomDataset(feature_vector,target,*list(L_time_slot.values()))
                # inputs = list(zip(feature_vector,target,*list(L_time_slot.values()) ))
                # sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=idr_torch.size,rank=idr_torch.rank,shuffle= ...)
                self.dataloader[training_mode] = DataLoader(inputs, 
                                                            batch_size=(feature_vector.size(0) if training_mode=='cal' else batch_size),
                                                            shuffle = (training_mode == 'train'),
                                                            #sampler=sampler,
                                                            num_workers=self.args.num_workers,
                                                            persistent_workers=self.args.persistent_workers,
                                                            pin_memory=self.args.pin_memory,
                                                            prefetch_factor=self.args.prefetch_factor,
                                                            drop_last=self.args.drop_last
                                                            ) 
            else:
                self.dataloader[training_mode] = None
            #self.dataloader[training_mode] = DataLoader(list(zip(feature_vector,target,L_time_slot)),batch_size=(feature_vector.size(0) if training_mode=='cal' else batch_size), shuffle = (True if ((training_mode == 'train') & self.shuffle ) else False)) 
        return(self.dataloader)

class MultiModelTrainer(object):
    def __init__(self,Datasets,model_list,dataloader_list,args,optimizer_list,loss_function,scheduler_list,args_embedding,dic_class2rpz,show_figure = True):
        super(MultiModelTrainer).__init__()
        trial_id1,trial_id2 = get_trial_id(args)
        self.trial_id1 = trial_id1
        self.trial_id2 = trial_id2
        self.Trainers = [Trainer(dataset,model,dataloader,args,optimizer,loss_function,scheduler,args_embedding,dic_class2rpz,fold = k,trial_id1 = self.trial_id1,trial_id2=self.trial_id2,show_figure= show_figure) for k,(dataset,dataloader,model,optimizer,scheduler) in enumerate(zip(Datasets,dataloader_list,model_list,optimizer_list,scheduler_list))]
        self.Loss_train =  torch.Tensor().to(args.device) #{k:[] for k in range(len(dataloader_list))}
        self.Loss_valid = torch.Tensor().to(args.device) #{k:[] for k in range(len(dataloader_list))}    
        self.alpha = args.alpha 
        self.picp = []   
        self.mpiw = []

    def K_fold_validation(self,mod_plot = 50,station = 0):
        results_by_fold = pd.DataFrame()
        for k,trainer in enumerate(self.Trainers):
            # Train valid model 
            if k == 0:
                print('\n')
            print(f"K_fold {k}")
            results_df = trainer.train_and_valid(mod = 10000,mod_plot = mod_plot,station = station)

            # Add Loss 
            self.Loss_train = torch.cat([self.Loss_train,torch.Tensor(trainer.train_loss).to(self.Loss_train).unsqueeze(0)],axis =  0) 
            self.Loss_valid = torch.cat([self.Loss_valid,torch.Tensor(trainer.valid_loss).to(self.Loss_valid).unsqueeze(0)],axis =  0) 
            # Testing
            if self.alpha is not None:
                preds,Y_true,_ = trainer.test_prediction(training_mode = 'test')
                pi = PI_object(preds,Y_true,self.alpha,type_calib = 'classic')
                self.picp.append(pi.picp)
                self.mpiw.append(pi.mpiw)
            

            results_df['fold'] = k
            results_df.to_csv(f"{save_folder}results/{trainer.trial_id}_results.csv")
            results_by_fold = pd.concat([results_by_fold,results_df])

        mean_picp = torch.Tensor(self.picp).mean()
        mean_mpiw = torch.Tensor(self.mpiw).mean() 
        assert len(self.Loss_train.mean(dim = 0)) == len(trainer.train_loss), 'Mean on the wrong axis'


        # On recupère une loss moyenne sur les K-fold. On prend le minimum atteint par cette moyenne.
        dict_scores,dict_last = {},{}
        for L_loss,name in zip([self.Loss_train,self.Loss_valid],['train_loss','valid_loss']):

            # Score: 
            score, indices = L_loss.min(dim = -1)
            score = score.mean()
            std_score = torch.Tensor([L_loss[k,i] for k,i in enumerate(indices)]).std()
            dict_scores.update({f"score_{name}": score.item(),
                                f"std_{name}": std_score.item()})
            # ...

            # Last Score: 
            last_score = L_loss.mean(dim = 0)[-1]
            last_std = L_loss.std(dim = 0)[-1]
            dict_last.update({f"last_{name}": last_score.item(),
                                f"last_std_{name}": last_std.item()})


        # ...

        return(results_by_fold,mean_picp,mean_mpiw,dict_last,dict_scores)#,mean_on_best_train_loss_by_fold,mean_on_best_valid_loss_by_fold)       


    

class Trainer(object):
        ## Trainer Classique pour le moment, puis on verra pour faire des Early Stop 
    def __init__(self,dataset,model,dataloader,args,optimizer,loss_function,scheduler = None,args_embedding  =None,dic_class2rpz = None, fold = None,trial_id1 = None,trial_id2 = None,show_figure = True):
        super().__init__()
        self.dataset = dataset
        self.dataloader = dataloader
        self.training_mode = 'train'
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.model = model 
        self.scheduler = scheduler
        self.scaler = GradScaler()
        self.train_loss = []
        self.valid_loss = []
        self.calib_loss =[]
        self.args = args
        self.alpha = args.alpha
        self.args_embedding = args_embedding
        #self.save_path  = f"best_model.pkl" if save_dir is not None else None
        self.fold = fold
        #self.save_dir = f"{save_dir}fold{fold}/" 
        self.best_valid = np.inf
        self.dic_class2rpz = dic_class2rpz
        self.picp_list = []
        self.mpiw_list = []
        self.show_figure = show_figure
        if trial_id1 is None:
            self.trial_id = get_trial_id(args,fold)
        else:
            self.trial_id = f"{trial_id1}{fold}{trial_id2}"
        if fold is not None:
            self.args.current_fold = fold

    def save_best_model(self,checkpoint,epoch,performance):
        ''' Save best model in .pkl format'''
        #update checkpoint
        checkpoint.update(epoch=epoch, state_dict=self.model.state_dict())
        save_best_model_and_update_json(checkpoint,self.trial_id,performance,self.args,save_dir = 'save/best_models/')
        # torch.save(checkpoint, f"{self.save_dir}{self.save_path}")   

    def plot_bokeh_and_save_results(self,results_df,epoch,station):
        Q = torch.zeros(1,next(iter(self.dataloader['test']))[0].size(1),1).to(self.args.device)  # Get Q null with shape [1,N,1]
        trial_save = f'latent_space_e{epoch}'
        pi,pi_cqr = generate_bokeh(self,self.dataloader,
                                    self.dataset,Q,self.args,self.dic_class2rpz,
                                    self.trial_id,
                                    trial_save,station = station,
                                    show_figure = self.show_figure,
                                    save_plot = True
                                    )
        valid_loss,train_loss = self.valid_loss[-1] if len(self.valid_loss)>0 else None, self.train_loss[-1] if len(self.train_loss)>0 else None
        if pi is None:
            picp,mpiw = None, None
        else:
            picp,mpiw = pi.picp,pi.mpiw

        dict_row = results2dict(self.args,epoch,picp,mpiw,valid_loss,train_loss)
        results_df = update_results_df(results_df,dict_row) 
        return(results_df)

    def train_valid_one_epoch(self):
        # Train and Valid each epoch 
        self.training_mode = 'train'
        self.model.train()   #Activate Dropout 
        self.loop()
        self.training_mode = 'validate'
        self.model.eval()   # Desactivate Dropout 
        self.loop() 

        # Update scheduler after each Epoch 
        self.chrono.torch_scheduler()
        self.update_scheduler()
        self.chrono.torch_scheduler()

        # Follow Update of Testing Metrics 
        self.chrono.track_pi()
        pi = self.track_pi()
        self.ray_tune_track(pi)
        self.chrono.track_pi()

    def update_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def display_usefull_information(self,epoch,mod,t0):
        if mod is not None:
            if epoch%mod==0:
                print(f"epoch: {epoch} \n min\epoch : {'{0:.2f}'.format((time.time()-t0)/60)}")
            if epoch == 1:
                print(f"Estimated time for training: {'{0:.1f}'.format(self.args.epochs*(time.time()-t0)/60)}min ")
    
    def get_pi_from_prediction(self):
        # Calibration 
        Q = self.conformal_calibration(self.args.alpha,self.dataset,
                                    conformity_scores_type = self.args.conformity_scores_type,
                                    quantile_method = self.args.quantile_method,print_info = False)  
        # Testing
        preds,Y_true,T_labels = self.test_prediction(training_mode = 'validate')
        #preds,Y_true,T_labels = self.testing(self,self.dataset, allow_dropout = False, training_mode = 'validate')

        # get PI
        pi = self.CQR_PI(preds,Y_true,self.args.alpha,Q,T_labels)

        return(pi)
    
    def track_pi(self):
        if self.args.track_pi:
            pi = self.get_pi_from_prediction()
            self.picp_list.append(pi.picp)
            self.mpiw_list.append(pi.mpiw)
            return(pi)
        else:
            return(None)


    def ray_tune_track(self,pi):
        if self.args.ray:
            if pi is not None:
                # Report usefull metrics
                report({"Loss_model" : self.valid_loss[-1], 
                            "MPIW" : pi.mpiw,
                            "PICP" : pi.picp}) 
            else:
                report({"Loss_model" : self.valid_loss[-1]})


    def train_and_valid(self,mod = None, mod_plot = None,station = 0):
        print(f'start training')
        checkpoint = {'epoch':0, 'state_dict':self.model.state_dict()}
        # Plot Init Latent Space and Accuracy (from random initialization) 
        if mod_plot is not None: 
            results_df = self.plot_bokeh_and_save_results(pd.DataFrame(),-1,station)
        else:
            results_df = None

        self.chrono = Chronometer()
        self.chrono.start()
        max_memory = 0
        for epoch in range(self.args.epochs):
            self.chrono.next_iter()
            t0 = time.time()
            # Train and Valid each epoch 
            self.train_valid_one_epoch()

            # Save best model (only if it's not a ray tuning)
            self.chrono.save_model()
            if (self.valid_loss[-1] < self.best_valid) & (not(self.args.ray)):
                self.best_valid = self.valid_loss[-1]
                performance = {'valid_loss': self.best_valid, 'epoch':epoch, 'training_over' : False}
                self.save_best_model(checkpoint,epoch,performance)
            self.chrono.save_model()


            # Plot Latent Space and get accuracy 
            self.chrono.plotting()
            self.display_usefull_information(epoch,mod,t0)
            if mod_plot is not None:
                if (epoch%mod_plot == 0)|(epoch== self.args.epochs -1):
                    results_df = self.plot_bokeh_and_save_results(results_df,(epoch+1),station)
            self.chrono.plotting()

            # Keep track on cpu-usage 
            max_memory = get_cpu_usage(max_memory)


        self.chrono.save_model()
        performance = {'valid_loss': self.best_valid, 'epoch':performance['epoch'], 'training_over' : True, 'fold': self.args.current_fold}
        self.save_best_model(checkpoint,epoch,performance)
        
        print(f'Training Throughput:{(self.args.epochs * len(self.dataset.df_verif_train))/np.sum(self.chrono.time_perf_train)} sequences per seconds')
        
        self.chrono.save_model()

        self.chrono.stop()
        self.chrono.display()
        print(print_memory_usage(max_memory))

        return(results_df)


    def loop(self):
        loss_epoch,nb_samples = 0,0
        if self.training_mode=='validation':
            self.chrono.validation()
        with torch.set_grad_enabled(self.training_mode=='train'):
            for x_b,y_b,*T_b in self.dataloader[self.training_mode]:
                t_b = T_b[self.args.calendar_class]
                x_b,y_b,t_b = x_b.to(self.args.device,non_blocking = self.args.non_blocking),y_b.to(self.args.device,non_blocking = self.args.non_blocking),t_b.to(self.args.device,non_blocking = self.args.non_blocking)

                #Forward 
                if self.training_mode=='train':
                    self.chrono.forward()

                if self.args.mixed_precision:
                    with autocast():
                        if self.args_embedding : 
                            pred = self.model(x_b,t_b.long())
                        else:
                            pred = self.model(x_b)
                        loss = self.loss_function(pred,y_b)
                else:
                    if self.args_embedding : 
                        pred = self.model(x_b,t_b.long())
                    else:
                        pred = self.model(x_b)
                    loss = self.loss_function(pred,y_b)       

                # Back propagation (after each mini-batch)
                if self.training_mode == 'train': 
                    self.chrono.backward()
                    loss = self.backpropagation(loss)

                # Keep track on metrics 
                nb_samples += x_b.shape[0]
                loss_epoch += loss.item()*x_b.shape[0]

                if self.training_mode == 'train': 
                    self.chrono.update()      
                    self.chrono.next_iter()

        if self.training_mode=='validation':
            self.chrono.validation()
        self.update_loss_list(loss_epoch,nb_samples,self.training_mode)

    def conformal_calibration(self,alpha,dataset,conformity_scores_type = 'max_residual',quantile_method = 'classic',print_info = True, calibration_calendar_class = None):
        ''' 
        Quantile estimator (i.e NN model) is trained on the proper set
        Conformity scores computed with quantile estimator on the calibration set
        And then the empirical th-quantile Q is computed with the conformity scores and quantile function

        inputs
        -------
        - alpha : is the miscoverage rate. such as  P(Y in C(X)) >= 1- alpha 
        - dataset : DataSet object. Allow us to unormalize tensor
        '''
        if calibration_calendar_class is None:
            calibration_calendar_class = self.args.calendar_class
        str_info = ''
        self.model.eval()
        with torch.no_grad():
            # Load Calibration Dataset :
            data = [[x_b,y_b,t_b[self.args.calendar_class],t_b[calibration_calendar_class]] for  x_b,y_b,*t_b in self.dataloader['cal']]
            X_cal,Y_cal,T_pred,T_cal = torch.cat([x_b for [x_b,_,_,_] in data]).to(self.args.device),torch.cat([y_b for [_,y_b,_,_] in data]).to(self.args.device),torch.cat([t_pred for [_,_,t_pred,_] in data]).to(self.args.device),torch.cat([t_cal for [_,_,_,t_cal] in data]).to(self.args.device)

            # Forward Pass: 
            if self.args_embedding : 
                preds = self.model(X_cal,T_pred.long())
            else:
                preds = self.model(X_cal) 

            if len(preds.size()) == 2:
                preds = preds.unsqueeze(1)
            # ...


            # get lower and upper band
            if preds.size(-1) == 2:
                lower_q,upper_q = preds[...,0].unsqueeze(-1),preds[...,1].unsqueeze(-1)   # The Model return ^q_l and ^q_u associated to x_b
        
            elif preds.size(-1) == 1:
                lower_q,upper_q = preds,preds 
            else:
                raise ValueError(f"Shape of model's prediction: {preds.size()}. Last dimension should be 1 or 2.")
            # ...
            
            # unormalized lower and upper band  
            lower_q, upper_q = dataset.unormalize_tensor(lower_q,device = self.args.device),dataset.unormalize_tensor(upper_q,device = self.args.device)
            Y_cal = dataset.unormalize_tensor(Y_cal,device = self.args.device)
            # ...

            # Get Confority scores: 
            if conformity_scores_type == 'max_residual':
                self.conformity_scores = torch.max(lower_q-Y_cal,Y_cal-upper_q).to(self.args.device) # Element-wise maximum        #'max(lower_q-y_b,y_b-upper_q)' is the quantile regression error function
            if conformity_scores_type == 'max_residual_plus_middle':
                str_info = str_info+ "\n|!| Conformity scores computation is not based on 'max(ql-y, y-qu)'"
                self.conformity_scores = torch.max(lower_q-Y_cal,Y_cal-upper_q) + ((lower_q>Y_cal)(upper_q<Y_cal))*(upper_q - lower_q)/2  # Element-wise maximum        #'max(lower_q-y_b,y_b-upper_q)' is the quantile regression error function
            # ...

            # Get Quantile:
            # If classic Calibration:
            if quantile_method == 'classic':  
                quantile_order = torch.Tensor([np.ceil((1 - alpha)*(X_cal.size(0)+1))/X_cal.size(0)]).to(self.args.device)
                #Q = torch.quantile(self.conformity_scores, quantile_order, dim = 0).to(self.device) #interpolation = 'higher'
                Q = get_higher_quantile(self.conformity_scores,quantile_order,device = self.args.device)
                output = Q

            # If Calibration by group of T_labels: 
            if quantile_method == 'compute_quantile_by_class':  # Calcul Higher Quantil for each calendar class. Several label can belongs to the same calendar class. The Quantile is computed through all residual of label of the same class
                #calendar_class = torch.cat([t_cal for [_,_,_,t_cal] in data])
                dic_label2Q = {}
            # ...


                # Compute quantile for each calendar class : 
                nb_label_with_quantile_1 = 0
                for label in T_cal.unique():
                    indices = torch.nonzero(T_cal == label,as_tuple = True)[0]
                    quantile_order = torch.Tensor([np.ceil((1 - alpha)*(indices.size(0)+1))/indices.size(0)]).to(self.args.device)  # Quantile for each class, so the quantile order is different as each class has a different length
                    quantile_order = min(torch.Tensor([1]).to(self.args.device),quantile_order)
                    if quantile_order == 1: 
                        nb_label_with_quantile_1 +=1
                        #print(f"label {label} has only {indices.size(0)} elements in his class. We then use quantile order = 1")
                    conformity_scores_i = self.conformity_scores[indices]
                    scores_counts = conformity_scores_i.size(0)
                    Q_i = get_higher_quantile(conformity_scores_i,quantile_order,device = self.args.device)
                    #Q_i = torch.quantile(conformity_scores_i, quantile_order, dim = 0)#interpolation = 'higher'
                    dic_label2Q[label.item()]= {'Q': Q_i,'count':scores_counts}

                str_info = str_info+ f"\nProportion of label with quantile order set to 1: {'{:.1%}'.format(nb_label_with_quantile_1/len(T_cal.unique()))}"
                output = dic_label2Q
        
        if print_info:
            print(str_info)

        return(output)
    
    def CQR_PI(self,preds,Y_true,alpha,Q,T_labels = None):
        pi = PI_object(preds,Y_true,alpha,type_calib = 'CQR',Q=Q,T_labels = T_labels)
        self.pi = pi
        return(pi)


    def backpropagation(self,loss):
        self.optimizer.zero_grad()
        if self.args.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        return(loss)
    
    def test_prediction(self,allow_dropout = False,training_mode = 'test',X = None, Y_true= None, T_labels= None):
        self.training_mode = training_mode
        if allow_dropout:
            self.model.train()
        else: 
            self.model.eval()
        with torch.no_grad():       
            if X is None:
                data = [[x_b,y_b,t_b[self.args.calendar_class]] for  x_b,y_b,*t_b in self.dataloader[training_mode]]
                X,Y_true,T_labels= torch.cat([x_b for [x_b,_,_] in data]).to(self.args.device),torch.cat([y_b for [_,y_b,_] in data]).to(self.args.device), torch.cat([t_b for [_,_,t_b] in data]).to(self.args.device)
            if self.args_embedding : 
                Pred = self.model(X,T_labels.long())
            else:
                Pred = self.model(X) 
                
        return(Pred,Y_true,T_labels)

    def testing(self,dataset, allow_dropout = False, training_mode = 'test',X = None, Y_true = None, T_labels = None): #metrics= ['mse','mae']
        (test_pred,Y_true,T_labels) = self.test_prediction(allow_dropout,training_mode,X,Y_true,T_labels)  # Get Normalized Pred and Y_true

        test_pred = dataset.unormalize_tensor(test_pred, device = self.args.device)
        Y_true = dataset.unormalize_tensor(Y_true, device = self.args.device)

        #df_metrics = evaluate_metrics(test_pred,Y_true,metrics)
        return(test_pred,Y_true,T_labels)#,df_metrics)  
    
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
    '''
    def __init__(self,df,init_df = None,mini= None, maxi = None, mean = None, normalized = False,time_step_per_hour = None,
                 train_df = None,cleaned_df = None,Weeks = None, Days = None, historical_len = None,step_ahead = None):
        self.length = len(df)
        self.df = df
        self.columns = df.columns
        self.normalized = normalized
        self.time_step_per_hour = time_step_per_hour
        self.df_dates = pd.DataFrame(self.df.index,index = np.arange(len(self.df)),columns = ['date'])
        self.train_df = train_df
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
        self.df_shifted = None
        self.invalid_indx_df = None
        self.remaining_dates = None
        self.time_slots_labels = None
        self.step_ahead = step_ahead
        self.Weeks = Weeks
        self.Days = Days
        self.historical_len = historical_len
        self.cleaned_df = cleaned_df

        
    def bijection_name_indx(self):
        colname2indx = {c:k for k,c in enumerate(self.columns)}
        indx2colname = {k:c for k,c in enumerate(self.columns)}
        return(colname2indx,indx2colname)
    
    def get_shift_between_set(self):
        shift_week = self.Weeks if self.Weeks is not None else 0
        shift_day = self.Days if self.Days is not None else 0
        self.shift_from_first_elmt = int(max(shift_week*24*7*self.time_step_per_hour,
                                shift_day*24*self.time_step_per_hour,
                                self.historical_len+self.step_ahead-1
                                ))
        self.shift_between_set = self.shift_from_first_elmt*timedelta(hours = 1/self.time_step_per_hour)


    def minmaxnorm(self,x,reverse = False):
        if reverse:
            x = x*(self.maxi - self.mini) +self.mini
        else :
            x = (x-self.mini)/(self.maxi-self.mini)
        return x   


    def minmax_normalize_df(self):

        self.mini = self.df_train.min()
        self.maxi = self.df_train.max()
        self.mean = self.df_train.mean()

        # Normalize : 
        normalized_df = self.minmaxnorm(self.df)  # Normalize the entiere dataset

        # Update state : 
        self.df = normalized_df
        self.normalized = True

    def normalize_df(self,minmaxnorm = True):
        assert self.normalized == False, 'Dataframe might be already normalized'
        if minmaxnorm:
            self.minmax_normalize_df()
        else:
            raise Exception('Normalization has not been coded')
        
    def unormalize_df(self,minmaxnorm):
        assert self.normalized == True, 'Dataframe might be already UN-normalized'
        if minmaxnorm:
            self.df = self.minmaxnorm(self.df,reverse = True)
        self.normalized = False

    def split_K_fold(self,args,invalid_dates):
        '''
        Split la DataSet Initiale en K-fold
        '''
        Datasets,DataLoader_list = [],[]
        #dic_class2rpz_list,dic_rpz2class_list,nb_words_embedding_list,time_slots_labels_list = [],[],[],[]
        # Récupère la df (On garde les valeurs interdite pour le moment, on les virera après. Il est important de les virer pour la normalisation, pour pas Normaliser la donnée avec des valeurs qui n'ont pas de sens.)
        df = self.df
        # Récupère la DataSet de Test Commune à tous: 
        dataset_init = DataSet(self.df, Weeks = self.Weeks, Days = self.Days, historical_len= self.historical_len,
                                   step_ahead=self.step_ahead,time_step_per_hour=self.time_step_per_hour)
        dataset_init.Dataset_save_folder = Dataset_get_save_folder(args,K_fold = 1,fold=0)
        data_loader_with_test,_,_,_,_ = dataset_init.split_normalize_load_feature_vect(args,invalid_dates,args.train_prop, args.valid_prop,args.test_prop)
        # Fait la 'Hold-Out' séparation, pour enlever les dernier mois de TesT
        df = df[: dataset_init.first_test_date]  

        # Récupère la Taille de cette DataFrame
        n = len(df)


        
        # Adapt Valid and Train Prop (cause we want Test_prop = 0)
        valid_prop_tmps = args.valid_prop/(args.train_prop+args.valid_prop)
        train_prop_tmps = args.train_prop/(args.train_prop+args.valid_prop)
        
        # Découpe la dataframe en K_fold 
        for k in range(args.K_fold):
            # Slicing 
            if args.validation == 'wierd_blocked':
                df_tmps = df[int((k/args.K_fold)*n):int(((k+1)/args.K_fold)*n)]

            if args.validation == 'sliding_window':
                width_dataset = int(n/(1+(args.K_fold-1)*valid_prop_tmps))   # Stay constant. W = N/(1 + (K-1)*Pv/(Pv+Pt))
                init_pos = int(k*valid_prop_tmps*width_dataset)    # Shifting of (valid_prop/train_prop)% of the width of the window, at each iteration 
                if k == args.K_fold - 1:
                    df_tmps = df[init_pos:]             
                else:
                    df_tmps = df[init_pos:init_pos+width_dataset]                   

            # On crée une DataSet à partir de df_tmps, qui a toujours la même taille, et toute les df_temps concaténée recouvre Valid Prop + Train Prop, mais pas Test Prop 
            dataset_tmps = DataSet(df_tmps, Weeks = self.Weeks, Days = self.Days, historical_len= self.historical_len,
                                   step_ahead=self.step_ahead,time_step_per_hour=self.time_step_per_hour)
            dataset_tmps.Dataset_save_folder = Dataset_get_save_folder(args,fold=k)
            if dataset_init.Weeks+dataset_init.historical_len+dataset_init.Days == 0:
                print(f"! H+D+W = {dataset_init.Weeks+dataset_init.historical_len+dataset_init.Days}, which mean the Tensor U will be set to a Null vector")

            data_loader,time_slots_labels,dic_class2rpz,dic_rpz2class,nb_words_embedding = dataset_tmps.split_normalize_load_feature_vect(args,invalid_dates,train_prop_tmps, valid_prop_tmps, 0)
            data_loader['test'] = data_loader_with_test['test']
            dataset_tmps.U_test, dataset_tmps.Utarget_test, dataset_tmps.time_slots_test, = dataset_init.U_test, dataset_init.Utarget_test, dataset_init.time_slots_test
            dataset_tmps.first_predicted_test_date,dataset_tmps.last_predicted_test_date = dataset_init.first_predicted_test_date,dataset_init.last_predicted_test_date
            dataset_tmps.first_test_date,dataset_tmps.last_test_date = dataset_init.first_test_date,dataset_init.last_test_date
            dataset_tmps.df_verif_test = dataset_init.df_verif_test
            dataset_tmps.df_test = dataset_init.df_test
             
            Datasets.append(dataset_tmps)
            DataLoader_list.append(data_loader)

        return(Datasets,DataLoader_list,time_slots_labels,dic_class2rpz,dic_rpz2class,nb_words_embedding)
    

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

    def shift_dates(self):
        # Weekkly periodic
        Dwt = [self.df_dates.shift((self.Weeks-i)*self.Week_nb_steps) for i in range(self.Weeks)] 
        # Daily periodic
        Ddt = [self.df_dates.shift((self.Days-i)*self.Day_nb_steps) for i in range(self.Days)] 
        # Recent Historic pattern 
        Dt = [self.df_dates.shift(self.step_ahead+(self.historical_len-i)) for i in range(1,self.historical_len+1)] 
        shifted_dates = Dwt+Ddt+Dt
        return(shifted_dates)

    def shift_values(self):
        # Weekkly periodic
        Uwt = [torch.unsqueeze(torch.Tensor(self.df.shift((self.Weeks-i)*self.Week_nb_steps).values),2) for i in range(self.Weeks)]
        # Daily periodic
        Udt = [torch.unsqueeze(torch.Tensor(self.df.shift((self.Days-i)*self.Day_nb_steps).values),2) for i in range(self.Days)]
         # Recent Historic pattern 
        Ut =  [torch.unsqueeze(torch.Tensor(self.df.shift(self.step_ahead+(self.historical_len-i)).values),2) for i in range(1,self.historical_len+1)]
        shifted_values = Uwt+Udt+Ut
        return(shifted_values)

    def remove_forbidden_prediction(self,invalid_dates):
        # Mask for dataframe df_verif
        df_shifted_forbiden = pd.concat([self.df_shifted[self.df_shifted[c].isin(invalid_dates)] for c in self.df_shifted.columns])  # Concat forbidden indexes within each columns
        forbidden_index = df_shifted_forbiden.index.unique()  # drop dupplicates

        # Mask for Tensor U, Utarget
        forbidden_indice_U = forbidden_index - self.shift_from_first_elmt  #shift index to get back to corresponding indices
        mask_U =  [e for e in np.arange(self.U.shape[0]) if e not in forbidden_indice_U]

        # Apply Mask
        self.U = self.U[mask_U]
        self.Utarget = self.Utarget[mask_U]
        self.df_verif = self.df_shifted.drop(forbidden_index)

    def get_df_shifted(self):
        # Get the shifted "Dates" of Feature Vector and Target
        shifted_dates = self.shift_dates()
        L_shifted_dates = shifted_dates + [self.df_dates]
        Names = [f't-{str(self.Week_nb_steps*(self.Weeks-w))}' for w in range(self.Weeks)] + [f't-{str(self.Day_nb_steps*(self.Days-d))}' for d in range(self.Days)] + [f't-{str(self.historical_len-t)}' for t in range(self.historical_len)]+ [f't+{self.step_ahead-1}']
        self.df_shifted = pd.DataFrame({name:lst['date'] for name,lst in zip(Names,L_shifted_dates)})[self.shift_from_first_elmt:] 

    def get_U_shifted(self):
        shifted_values = self.shift_values()
        self.Utarget = torch.unsqueeze(torch.Tensor(self.df.values),2)[self.shift_from_first_elmt:]

        try:
            self.U = torch.cat(shifted_values,dim=2)[:][self.shift_from_first_elmt:]
        except:
            assert self.Weeks+self.historical_len+self.Days == 0, 'something is going wrong with the previous line'
            self.U = self.Utarget*0


    def get_feature_vect(self): 
        # Get shifted Feature Vector and shifted Target
        self.get_U_shifted()       
        # Get the df of associated TimeStamp of the shifter Feature Vector and shifted Target
        self.get_df_shifted()

    def split_tensors(self):
        ''' Split tensor U in Train/Valid/Test part '''
        self.U_train = self.U[self.first_train_U:self.last_train_U]
        self.U_valid = self.U[self.first_valid_U:self.last_valid_U] if self.first_valid_U is not None else None   
        self.U_test = self.U[self.first_test_U:self.last_test_U] if self.first_test_U is not None else None

        self.Utarget_train = self.Utarget[self.first_train_U:self.last_train_U] 
        self.Utarget_valid = self.Utarget[self.first_valid_U:self.last_valid_U] if self.first_valid_U is not None else None
        self.Utarget_test = self.Utarget[self.first_test_U:self.last_test_U] if self.first_test_U is not None else None

        if self.time_slots_labels is not None : 
            self.time_slots_train = {calendar_class: self.time_slots_labels[calendar_class][self.first_train_U:self.last_train_U] for calendar_class in range(len(self.nb_class)) }
            self.time_slots_valid = {calendar_class: self.time_slots_labels[calendar_class][self.first_valid_U:self.last_valid_U] if self.first_valid_U is not None else None for calendar_class in range(len(self.nb_class))}
            self.time_slots_test = {calendar_class: self.time_slots_labels[calendar_class][self.first_test_U:self.last_test_U] if self.first_test_U is not None else None for calendar_class in range(len(self.nb_class)) }
            #self.time_slots_valid = self.time_slots_labels[self.args.calendar_class][self.first_valid_U:self.last_valid_U] if self.first_valid_U is not None else None
            #self.time_slots_test = self.time_slots_labels[self.args.calendar_class][self.first_test_U:self.last_test_U] if self.first_test_U is not None else None    

    def train_valid_test_split(self,train_prop,valid_prop,test_prop,time_slots_labels = None):
        # Split with iterative method 
        split_path = f"{self.Dataset_save_folder}split_limits.pkl" 
        if os.path.exists(split_path):
            try:
                split_limits = read_object(split_path)
            except:
                split_limits= train_valid_test_split_iterative_method(self,self.df_verif,train_prop,valid_prop,test_prop)
                save_object(split_limits, split_path)
                print(f"split_limits.pkl has never been saved or issue with last .pkl save")
        else : 
            split_limits= train_valid_test_split_iterative_method(self,self.df_verif,train_prop,valid_prop,test_prop)
            save_object(split_limits, split_path)

        first_predicted_train_date= split_limits['first_predicted_train_date']
        last_predicted_train_date = split_limits['last_predicted_train_date']
        first_predicted_valid_date = split_limits['first_predicted_valid_date']
        last_predicted_valid_date = split_limits['last_predicted_valid_date']
        first_predicted_test_date = split_limits['first_predicted_test_date']
        last_predicted_test_date = split_limits['last_predicted_test_date']

        # Keep track on predicted limits (dates): 
        self.first_predicted_train_date,self.last_predicted_train_date = first_predicted_train_date,last_predicted_train_date
        self.first_predicted_valid_date,self.last_predicted_valid_date = first_predicted_valid_date,last_predicted_valid_date
        self.first_predicted_test_date,self.last_predicted_test_date = first_predicted_test_date,last_predicted_test_date

        # Keep track on DataFrame Verif:
        predicted_dates = self.df_verif.iloc[:,-1]
        self.df_verif_train = self.df_verif[( predicted_dates >= self.first_predicted_train_date) & (predicted_dates < self.last_predicted_train_date)]
        self.df_verif_valid = self.df_verif[(predicted_dates >= self.first_predicted_valid_date) & (predicted_dates < self.last_predicted_valid_date)] if self.last_predicted_valid_date is not None else None
        self.df_verif_test = self.df_verif[(predicted_dates >= self.first_predicted_test_date) & (predicted_dates < self.last_predicted_test_date)]  if self.last_predicted_test_date is not None else None

        # Keep track on DataFrame Limits (dates): 
        self.first_train_date,self.last_train_date = self.df_verif_train.iat[0,0] ,self.df_verif_train.iat[-1,-1]
        if valid_prop > 1e-3 :
            self.first_valid_date,self.last_valid_date = self.df_verif_valid.iat[0,0] ,self.df_verif_valid.iat[-1,-1]
        else:
            self.first_valid_date,self.last_valid_date = None, None

        if test_prop > 1e-3 :
            self.first_test_date,self.last_test_date = self.df_verif_test.iat[0,0] ,self.df_verif_test.iat[-1,-1]
        else:
            self.first_test_date,self.last_test_date = None, None
        
        # Get All the involved dates and keep track on splitted DataFrame 
        self.df_train = self.df.reindex(self.df_verif_train.stack().unique())
        self.df_valid = self.df.reindex(self.df_verif_valid.stack().unique()) if valid_prop > 1e-3 else None
        self.df_test = self.df.reindex(self.df_verif_test.stack().unique()) if test_prop > 1e-3 else None

        # Get all the limits for U / Utarget split : 
        self.first_train_U = self.df_verif.index.get_loc(self.df_verif[self.df_verif[f"t+{self.step_ahead - 1}"] == self.first_predicted_train_date].index[0])
        self.last_train_U = self.df_verif.index.get_loc(self.df_verif[self.df_verif[f"t+{self.step_ahead - 1}"] == self.last_predicted_train_date].index[0])

        self.first_valid_U = self.df_verif.index.get_loc(self.df_verif[self.df_verif[f"t+{self.step_ahead - 1}"] == self.first_predicted_valid_date].index[0]) if valid_prop > 1e-3 else None
        self.last_valid_U = self.df_verif.index.get_loc(self.df_verif[self.df_verif[f"t+{self.step_ahead - 1}"] == self.last_predicted_valid_date].index[0]) if valid_prop > 1e-3 else None

        self.first_test_U = self.df_verif.index.get_loc(self.df_verif[self.df_verif[f"t+{self.step_ahead - 1}"] == self.first_predicted_test_date].index[0]) if test_prop > 1e-3 else None
        self.last_test_U = self.df_verif.index.get_loc(self.df_verif[self.df_verif[f"t+{self.step_ahead - 1}"] == self.last_predicted_test_date].index[0]) if test_prop > 1e-3 else None

    def split_normalize_load_feature_vect(self,args,invalid_dates,train_prop,valid_prop,test_prop
                                          #,calib_prop,batch_size,calendar_class
                                          ):
        self.get_shift_between_set()   # get shift indice and shift date from the first element / between each dataset 
        self.get_feature_vect()  # Build 'df_shifted'.
        self.remove_forbidden_prediction(invalid_dates) # Build 'df_verif' , which is df_shifted without sequences which contains invalid date

        # Get Index to Split df, U, Utarget, time_slots_labels
        self.train_valid_test_split(train_prop,valid_prop,test_prop)  # Create df_train,df_valid,df_test, df_verif_train, df_verif_valid, df_verif_test, and dates limits for each df and each tensor U

        # Normalize 
        self.normalize_df(minmaxnorm = True)   # Normalize dataset.df thank to dataset.train_df

        # Re-load U  and U_target, while df is now correctly normalized
        self.get_feature_vect()  # Build 'df_shifted'.
        self.remove_forbidden_prediction(invalid_dates) # Build 'df_verif' , which is df_shifted without sequences which contains invalid date

        # get Associated time_slots_labels >
        time_slots_labels,dic_class2rpz,dic_rpz2class,nb_words_embedding = get_time_slots_labels(self)

        # Split U in  U_train, U_valid, U_test thanks to 'df_verif' and the date limits of the df_train/df_valid/df_test
        self.split_tensors()

        #   DataLoader 
        data_loader_obj = DictDataLoader(self,args)
        data_loader = data_loader_obj.get_dictdataloader(args.batch_size)
        return(data_loader,time_slots_labels,dic_class2rpz,dic_rpz2class,nb_words_embedding)

    # =====================================================================================================================================
    # Probablement à supprimer : 

    def remove_invalid_dates(self,invalid_dates):
        if invalid_dates is not None:
            invalid_dates = invalid_dates.intersection(self.df.index)
            tmps_df = self.df.drop(invalid_dates)
        else:
            tmps_df = self.df

        self.remaining_dataset = tmps_df

    def shift_data(self):

        # Weekkly periodic
        Uwt = [torch.unsqueeze(torch.Tensor(self.df.shift((self.Weeks-i)*self.Week_nb_steps).values),2) for i in range(self.Weeks)]
        Dwt = [self.df_dates.shift((self.Weeks-i)*self.Week_nb_steps) for i in range(self.Weeks)] 

        # Daily periodic
        Udt = [torch.unsqueeze(torch.Tensor(self.df.shift((self.Days-i)*self.Day_nb_steps).values),2) for i in range(self.Days)]
        Ddt = [self.df_dates.shift((self.Days-i)*self.Day_nb_steps) for i in range(self.Days)] 

        # Recent Historic pattern 
        Ut =  [torch.unsqueeze(torch.Tensor(self.df.shift(self.step_ahead+(self.historical_len-i)).values),2) for i in range(1,self.historical_len+1)]
        Dt = [self.df_dates.shift(self.step_ahead+(self.historical_len-i)) for i in range(1,self.historical_len+1)] 

        shifted_values = Uwt+Udt+Ut
        shifted_dates = Dwt+Ddt+Dt

        return(shifted_values,shifted_dates)

    def get_invalid_indx(self,invalid_dates:list):
        '''invalid_dates:  list of Tmestamp dates 
        from a list of dates, return 'invalid_indices_tensor', which correspond to the forbidden indices in the Tensor
        and return 'invalid_indx_df' which correspond to the forbidden index within the dataframe (where the first index can be > 0)'''
        # Get all the row were the invalid dates are used 
        df_verif_forbiden = pd.concat([self.df_shifted[self.df_shifted[c].isin(invalid_dates)] for c in self.df_shifted.columns])

        # Get the associated index 
        self.invalid_indx_df = df_verif_forbiden.index

        # Shift them in relation to the tensor 
        self.invalid_indices_tensor = self.invalid_indx_df - self.shift_from_first_elmt

        return(self.invalid_indices_tensor,self.invalid_indx_df)
    
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
        self.remaining_dates = self.df_shifted.loc[selected_dates_index,[f't+{self.step_ahead-1}']]
        return(self.U,self.Utarget,self.remaining_dates)





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