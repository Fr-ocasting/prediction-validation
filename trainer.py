import torch 
import pandas as pd 
import numpy as np 
import time
from torch.cuda.amp import autocast,GradScaler
#if torch.cuda.is_available():
#    torch.backends.cuda.matmul.allow_tf32 = True
#    torch.backends.cudnn.allow_tf32  = True

try :
    from plotting.plotting_bokeh import generate_bokeh
except:
    print('no plotting bokeh available')
from profiler.chrono import Chronometer

import pkg_resources

try: 
    from ray import tune,train
    ray_version = pkg_resources.get_distribution("ray").version
    if ray_version.startswith('2.7') or ray_version.startswith('3'):
        report = train.report
    else:
        report = tune.report
except : 
    print('Training and Hyper-parameter tuning with Ray is not possible')

from profiler.profiler import print_memory_usage,get_cpu_usage
from utils.save_results import results2dict, update_results_df, save_best_model_and_update_json,get_trial_id
from utils.metrics import evaluate_metrics
from utils.utilities import load_inputs_from_dataloader
from PI.PI_object import PI_object
from PI.PI_calibration import Calibrator
from constants.paths import SAVE_DIRECTORY



class Trainer(object):
        ## Trainer Classique pour le moment, puis on verra pour faire des Early Stop 
    def __init__(self,dataset,model,args,optimizer,loss_function,scheduler = None,dic_class2rpz = None, fold = None,trial_id = None,show_figure = False,save_folder =None):
        super().__init__()
        self.bool_contextual_data = len(dataset.contextual_tensors)>0
        self.nb_train_seq = len(dataset.tensor_limits_keeper.df_verif_train)


        self.dataloader = dataset.dataloader
        self.training_mode = 'train'
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.out_dim_factor = dataset.out_dim_factor
        self.step_ahead = dataset.step_ahead

        if args.loss_function_type == 'MSE':
            self.metrics = ['mse','mae','mape','mase'] 
            self.type_calib = None
            self.alpha = None
            args.track_pi = False
        elif args.loss_function_type == 'quantile':
            self.metrics = ['PICP','MPIW'] 
            self.type_calib = args.type_calib
            self.alpha = args.alpha
        else:
            raise NotImplementedError(f"metrics associated to {args.loss_function_type} has not been implemented")
        
        if args.torch_compile == 'compile':
            self.model = torch.compile(model,
                                       fullgraph = False,
                                       backend = args.backend,
                                       mode = None
                                        )
        elif args.torch_compile == 'jit_script':
            self.model = torch.jit.script(model)
        elif (args.torch_compile is None) or not(args.torch_compile):
            self.model = model 
        else:
            raise NotImplementedError(f"torch compile {args.torch_compile} has not been implemented")

        self.scheduler = scheduler
        if args.mixed_precision:
            self.scaler = GradScaler()
        self.train_loss = []
        self.valid_loss = []

        self.args = args
        
        self.fold = fold
        self.best_valid = np.inf
        self.dic_class2rpz = dic_class2rpz
        self.picp_list = []
        self.mpiw_list = []
        self.calib_loss =[]
        self.show_figure = show_figure

        if hasattr(args,'keep_best_weights') and (args.keep_best_weights):
            self.keep_best_weights =  args.keep_best_weights
        else:
            self.keep_best_weights = False

        if trial_id is None:
            self.trial_id = get_trial_id(args) 
        else:
            self.trial_id = trial_id
        if fold is not None: 
            self.trial_id = f"{self.trial_id}_f{fold}"
            self.args.current_fold = fold
        else:
            self.trial_id = f"{self.trial_id}_f{-1}"
            
        
        if save_folder is not None:
            self.best_model_save_directory = f"{SAVE_DIRECTORY}/{save_folder}/best_models"
        else:
            self.best_model_save_directory = f"{SAVE_DIRECTORY}/best_models"
            


    def save_best_model(self,checkpoint,epoch,performance,update_checkpoint = True):
        ''' Save best model in .pkl format'''
        #update checkpoint
        checkpoint.update(epoch=epoch, state_dict=self.model.state_dict())

        if self.keep_best_weights:
            self.best_weights = self.model.state_dict()
        save_best_model_and_update_json(checkpoint,self.trial_id,performance,self.args,
                                        save_dir = self.best_model_save_directory,
                                        update_checkpoint = update_checkpoint)

    def plot_bokeh_and_save_results(self,normalizer,df_verif_test,results_df,epoch,station):
        Q = torch.zeros(1,next(iter(self.dataloader['test']))[0].size(1),1).to(self.args.device)  # Get Q null with shape [1,N,1]
        trial_save = f'latent_space_e{epoch}'

        pi,pi_cqr = generate_bokeh(self,normalizer,
                                   df_verif_test,Q,self.args,
                                   self.trial_id,
                                   trial_save,
                                   station=station,
                                   show_figure = self.show_figure,
                                   save_plot = True)

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
        self.loop_epoch()

        self.training_mode = 'valid'
        self.model.eval()   # Desactivate Dropout 
        Preds,Y_true,T_labels = self.loop_epoch() 

        # Update scheduler after each Epoch 
        self.chrono.torch_scheduler()
        self.update_scheduler()
        self.chrono.torch_scheduler()

        # Follow Update of Testing Metrics 
        self.chrono.track_pi()
        pi = self.track_pi(Preds,Y_true,T_labels)
        self.ray_tune_track(pi)
        self.chrono.track_pi()

    def update_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def display_usefull_information(self,epoch,mod,t0):
        if epoch == 1:
            print(f"epoch: {epoch} \n min\epoch : {'{0:.2f}'.format((time.time()-t0)/60)}")
        if epoch == 2:
            print(f"Estimated time for training: {'{0:.1f}'.format(self.args.epochs*(time.time()-t0)/60)}min ")
    
    def get_pi_from_prediction(self,Preds,Y_true,T_labels):
        if self.type_calib == 'CQR':
            # Get Quantile tensor from Calibration with 'cal' datalaoder:
            Q = self.conformal_calibration(self.alpha,
                                        conformity_scores_type = self.args.conformity_scores_type,
                                        quantile_method = self.args.quantile_method,print_info = False)  
            # get PI from Prediction and apply the Calibration :
            pi = self.CQR_PI(Preds,Y_true,self.args.alpha,Q,T_labels)
            # ....
        elif self.type_calib == 'classic':
            pi = PI_object(Preds,Y_true,type_calib = 'classic')   
        else:
            raise NotImplementedError(f'Type of calibration {self.type_calib} for the quantile regression has not been implemented')

        return(pi)
    
    def track_pi(self,Preds,Y_true,T_labels):
        if self.args.track_pi:
            pi = self.get_pi_from_prediction(Preds,Y_true,T_labels)
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

    def prefetch_to_device(self,loader):
        if loader is not None :
            return [(x.to(self.args.device, non_blocking=self.args.non_blocking), y.to(self.args.device, non_blocking=self.args.non_blocking), [t.to(self.args.device, non_blocking=self.args.non_blocking) for t in T])
            for x, y, *T in loader]
        else :
            return None
    
    def train_and_valid(self,normalizer = None,df_verif_test = None,mod = None, mod_plot = None,station = 0):
        print(f'\nstart training')
        checkpoint = {'epoch':0, 'state_dict':self.model.state_dict()}

        # Plot Init Latent Space and Accuracy (from random initialization) 
        if mod_plot is not None: 
            results_df = self.plot_bokeh_and_save_results(normalizer,df_verif_test,pd.DataFrame(),-1,station)
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
            if (self.valid_loss[-1] < self.best_valid) & (not(self.args.ray)):
                self.chrono.save_model()
                self.best_valid = self.valid_loss[-1]
                self.performance = {'valid_loss': self.best_valid, 'epoch':epoch, 'training_over' : False}

                # Keep Track on Test Metrics
                Preds_test,Y_true_test,_ = self.test_prediction(allow_dropout = False,training_mode = 'test',track_loss=False,normalizer=normalizer)
                test_metrics = evaluate_metrics(Preds_test,Y_true_test,metrics = self.metrics,
                                                 alpha = self.alpha, type_calib = self.type_calib,dic_metric = {})
                self.performance.update({'test_metrics': test_metrics})

                # Keep Track on Valid Metrics:
                Preds_valid,Y_true_valid,_ = self.test_prediction(allow_dropout = False,training_mode = 'valid',track_loss=False,normalizer=normalizer)
                valid_metrics = evaluate_metrics(Preds_valid,Y_true_valid,metrics = self.metrics,
                                                 alpha = self.alpha, type_calib = self.type_calib,dic_metric = {})
                self.performance.update({'valid_metrics': valid_metrics})    
                # ...

                self.save_best_model(checkpoint,epoch,self.performance)
                self.chrono.save_model()


            # Plot Latent Space and get accuracy 
            self.chrono.plotting()
            self.display_usefull_information(epoch,mod,t0)
            if mod_plot is not None:
                if (epoch%mod_plot == 0)|(epoch== self.args.epochs -1):
                    results_df = self.plot_bokeh_and_save_results(normalizer,df_verif_test,results_df,(epoch+1),station)
            self.chrono.plotting()

            # Keep track on cpu-usage 
            max_memory = get_cpu_usage(max_memory)

        # Allow to keep track on the final metrics, and if the trial is 'terminated'
        if (not(self.args.ray)):
            self.chrono.save_model()
            if hasattr(self,'performance'):
                self.performance = {'valid_loss': self.best_valid,
                                    'valid_metrics':self.performance['valid_metrics'], 
                                    'test_metrics':self.performance['test_metrics'],
                                    'epoch':self.performance['epoch'], 
                                    'training_over' : True, 
                                    'fold': self.args.current_fold
                                    }
                                    
                self.save_best_model(checkpoint,epoch,self.performance,update_checkpoint = False)
                print(f"\nTraining Throughput:{'{:.2f}'.format((self.args.epochs * self.nb_train_seq)/np.sum(self.chrono.time_perf_train))} sequences per seconds")
            self.chrono.save_model()

        self.chrono.stop()
        self.chrono.display()
        print(print_memory_usage(max_memory))
        torch.cuda.empty_cache()

        return(results_df)
    
    '''
    def prefetch(self):
        self.chrono.prefetch_all_data()
        if hasattr(self,'already_prefetch'):
            self.dataset.prefetch_train_loader = self.prefetch_to_device(self.dataloader['train'])
        else:
            self.dataset.prefetch_train_loader = self.prefetch_to_device(self.dataloader['train']) #if hasattr(self.dataset,'train_loader'): 
            self.dataset.prefetch_valid_loader = self.prefetch_to_device(self.dataloader['valid']) #if hasattr(self.dataset,'valid_loader'): 
            self.dataset.prefetch_test_loader = self.prefetch_to_device(self.dataloader['test']) #if hasattr(self.dataset,'test_loader'): 
            self.dataset.prefetch_cal_loader = self.prefetch_to_device(self.dataloader['cal']) #if hasattr(self.dataset,'cal_loader'): 
            self.already_prefetch = True
        self.chrono.prefetch_all_data()
    '''
            
    def get_loader(self):
        if self.args.prefetch_all:
            raise ValueError('prefetch_all bizarre, à re-coder')
            #loader = self.dataloader_gpu[self.training_mode]
            if self.training_mode == 'train': loader = self.dataset.prefetch_train_loader
            if self.training_mode == 'valid': loader = self.dataset.prefetch_valid_loader
            if self.training_mode == 'test': loader = self.dataset.prefetch_test_loader
            if self.training_mode == 'cal': loader = self.dataset.prefetch_cal_loader           
        else:
            if self.training_mode == 'train': loader = self.dataloader['train']
            if self.training_mode == 'valid': loader = self.dataloader['valid']
            if self.training_mode == 'test': loader = self.dataloader['test']
            if self.training_mode == 'cal': loader = self.dataloader['cal']
        return(loader)
    
    def load_to_device(self,x_b,y_b,contextual_b): # T_b
        # If not already Pre-fetch: 
        if not(self.args.prefetch_all):
            #t_b = T_b[self.args.calendar_class]
            x_b = x_b.to(self.args.device,non_blocking = self.args.non_blocking)
            y_b = y_b.to(self.args.device,non_blocking = self.args.non_blocking)
            if contextual_b is not None: 
                contextual_b = [c_b.to(self.args.device,non_blocking = self.args.non_blocking) for c_b in contextual_b]
            
        return(x_b,y_b,contextual_b)
    
    def loop_batch(self,x_b,y_b,contextual_b,nb_samples,loss_epoch):
        #print('position of contextual data: ',self.args.contextual_positions)
        #print('contextual_b: ',[c.size() for c in contextual_b])
        
        #Forward 
        if self.training_mode=='train':
            self.chrono.forward()
        # Plus clean : with autocast(enabled = self.args.mixed_precision):
        if self.args.mixed_precision:
                with autocast():  #dtype=torch.bfloat16
                    pred = self.model(x_b,contextual_b)
                    loss = self.loss_function(pred.float(),y_b)
        else:
            pred = self.model(x_b,contextual_b)
            loss = self.loss_function(pred.float(),y_b)
        #print('loss: ',loss)
        #print('pred: ', pred.dtype, pred.size())
        #print('y_b: ', y_b.dtype, y_b.size())
        #print(self.loss_function)

        # Back propagation (after each mini-batch)
        if self.training_mode == 'train': 
            self.chrono.backward()
            loss = self.backpropagation(loss)
            #print('pred: ', pred.dtype, pred.size())
            #print('y_b: ', y_b.dtype, y_b.size())
            #print(self.loss_function)

        # Keep track on metrics 
        nb_samples += x_b.shape[0]
        loss_epoch += loss.item()*x_b.shape[0]

        if self.training_mode == 'train': 
            self.chrono.update()      
            self.chrono.next_iter()

        if 'calibration_calendar' in self.args.contextual_positions.keys(): 
            t_label =  contextual_b[self.args.contextual_positions['calibration_calendar']]
        else: 
            t_label = None
            
        return(pred,y_b,t_label,nb_samples,loss_epoch)
        
    def loop_through_batches(self,loader):
        ''' Small difference whether we first prefetch or not (T_b or *T_b) '''
        nb_samples,loss_epoch = 0,0
        Preds,Y_true,T_labels = [],[],[]

        if self.args.prefetch_all:
            raise ValueError('prefetch_all not correctly implemented')

        for inputs in loader:
            if self.bool_contextual_data:
                x_b,y_b,contextual_b = inputs
            else:
                x_b,y_b = inputs
                contextual_b = None

            x_b,y_b,contextual_b = self.load_to_device(x_b,y_b,contextual_b)
            pred,y_b,t_label,nb_samples,loss_epoch = self.loop_batch(x_b,y_b,contextual_b,nb_samples,loss_epoch)

            Preds.append(pred)
            Y_true.append(y_b)
            T_labels.append(t_label)

        # Torch concat : 
        Preds = torch.cat(Preds)
        Y_true = torch.cat(Y_true)
        if t_label is not None: T_labels = torch.cat(T_labels)
        # ... 
    
        return(Preds,Y_true,T_labels,nb_samples,loss_epoch)

    def loop_epoch(self,track_loss=True):
        if self.training_mode=='valid': 
            if hasattr(self,'chrono'):
                self.chrono.validation()
        #if self.training_mode=='cal': self.chrono.calibration()
        

        if (self.args.prefetch_all) & (self.training_mode=='train'):
            raise NotImplementedError('Prefetch all is not correctly imlpemented. If you do, take care not to overload the object trainer with too many elements.')
            self.prefetch()
            
        with torch.set_grad_enabled(self.training_mode=='train'):
            loader = self.get_loader()
            Preds,Y_true,T_labels,nb_samples,loss_epoch = self.loop_through_batches(loader)

            if self.training_mode == 'train':
                self.gradient_tracking()


        if self.training_mode=='valid':
            if hasattr(self,'chrono'):
                self.chrono.validation()
        #if self.training_mode=='cal': self.chrono.calibration()

        if track_loss:
            self.update_loss_list(loss_epoch,nb_samples,self.training_mode)
        return Preds,Y_true,T_labels

    def conformal_calibration(self,alpha,conformity_scores_type = 'max_residual',quantile_method = 'classic',print_info = True, calibration_calendar_class = None):
        ''' 
        Quantile estimator (i.e NN model) is trained on the proper set
        Conformity scores computed with quantile estimator on the calibration set
        And then the empirical th-quantile Q is computed with the conformity scores and quantile function

        inputs
        -------
        - alpha : is the miscoverage rate. such as  P(Y in C(X)) >= 1- alpha 
        '''
        # Load calibrator
        calibrator = Calibrator(alpha,self.args.device)

        # Predict : set attributes Preds, Y_cal, T_cal
        calibrator.get_prediction(self)

        # Lower / Upper band 
        calibrator.get_lower_upper_bands()
        calibrator.unormalize(self)

        # Get conformity scores
        calibrator.get_conformity_scores(conformity_scores_type)

        # Get calendar labels to compute calibration by group of time-slot
        #calibrator.get_calendar_label_for_grouped_calibration(self,self.args.contextual_positions)

        # Get Quatile Tensor
        calibrator.get_quantile_tensor(quantile_method)
        return(calibrator.Q)
    
    def CQR_PI(self,preds,Y_true,alpha,Q,T_labels = None):
        pi = PI_object(preds,Y_true,alpha,type_calib = 'CQR',Q=Q,T_labels = T_labels)
        return(pi)

    def backpropagation(self,loss):
        self.optimizer.zero_grad()
        # Useless if autocast is used with torch.bfloat16
        if True:
            if self.args.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
        else:
            loss.backward()
            self.optimizer.step()
        return(loss)
    
    def load_all_inputs_from_training_mode(self,training_mode):
        X,Y,X_c,nb_contextual = load_inputs_from_dataloader(self.dataloader[training_mode],self.args.device)
        return X,Y,X_c,nb_contextual
    
    def test_prediction(self,allow_dropout = False,training_mode = 'test',X = None, Y_true= None, T_labels= None,track_loss = False,normalizer = None):
        self.training_mode = training_mode
        if allow_dropout:
            self.model.train()
        else: 
            self.model.eval()

        with torch.no_grad():
            Preds,Y_true,T_labels = self.loop_epoch(track_loss)
        
        if normalizer is not None: 
            Preds = Preds.detach().cpu()
            Y_true = Y_true.detach().cpu()
            Preds = normalizer.unormalize_tensor(inputs = Preds,feature_vect = True) #  device = self.args.device
            Y_true = normalizer.unormalize_tensor(inputs = Y_true,feature_vect = True) # device = self.args.device

        return(Preds,Y_true,T_labels)

    def testing(self,normalizer, allow_dropout = False, training_mode = 'test',X = None, Y_true = None, T_labels = None,track_loss = False): #metrics= ['mse','mae']
        (Preds,Y_true,T_labels) = self.test_prediction(allow_dropout,training_mode,X,Y_true,T_labels,track_loss)  # Get Normalized Pred and Y_true
        Preds = Preds.detach().cpu()
        Y_true = Y_true.detach().cpu()
        T_labels = [tensor_ohe.detach().cpu() if tensor_ohe is not None else None for tensor_ohe in T_labels]

        # Set feature_vect = True cause output last dimension = 2 if quantile_loss or = 1.
        '''normalizer = self.dataset.normalizer.unormalize_tensor'''
        if normalizer is not None:
            Preds = normalizer.unormalize_tensor(inputs = Preds,feature_vect = True) #  device = self.args.device
            Y_true = normalizer.unormalize_tensor(inputs = Y_true,feature_vect = True) # device = self.args.device
        return(Preds,Y_true,T_labels)
    def update_loss_list(self,loss_epoch,nb_samples,training_mode):
        if training_mode == 'train':
            self.train_loss.append(loss_epoch/nb_samples)
        elif training_mode == 'valid':
            self.valid_loss.append(loss_epoch/nb_samples)
        elif training_mode == 'cal':
            self.calib_loss.append(loss_epoch/nb_samples)
    def gradient_tracking(self):
        module_list = ['te','core_model','output_module','netmob_vision']

        if not(hasattr(self,'gradient_metrics')):  
            self.gradient_metrics = {name:{} for name in module_list}  
        with torch.no_grad():
            for name in module_list:
                module = getattr(self.model, name, None)
                if module is not None:
                    grads = [p.grad for p in module.parameters() if p.grad is not None]
                    if grads:
                        all_grads = torch.cat([g.flatten() for g in grads])
                        metrics = {
                            #'max_grad': all_grads.abs().max().item(),
                            'abs_mean_grad': all_grads.abs().mean().item(),
                            'abs_median_grad': all_grads.abs().median().item(),
                            'median_grad': all_grads.median().item(),
                            'Q25': torch.quantile(all_grads,0.25).item(),
                            'Q75': torch.quantile(all_grads,0.75).item(),
                        }

                        for metric in metrics.keys(): 
                            if not(metric in self.gradient_metrics[name].keys()): 
                                self.gradient_metrics[name][metric] = []  

                        for metric in metrics.keys():
                            self.gradient_metrics[name][metric].append(metrics[metric])

        

