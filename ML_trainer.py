import torch 
import pandas as pd 
import numpy as np 
import time
from torch.cuda.amp import autocast,GradScaler
from datetime import datetime
if torch.cuda.is_available():
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32  = True

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
from utils.save_results import get_trial_id

from constants.paths import SAVE_DIRECTORY

from torchinfo import summary
from profiler.profiler import model_memory_cost
from dl_models.SARIMAX.SARIMAX import SARIMAX
from dl_models.XGBoost.XGBoost import XGBoost

class ML_trainer(object):
        ## Trainer Classique pour le moment, puis on verra pour faire des Early Stop 
    def __init__(self,dataset,args,optimizer,loss_function,scheduler = None,dic_class2rpz = None, fold = None,trial_id = None,show_figure = False,save_folder =None):
        super().__init__()
        self.bool_contextual_data = len(dataset.contextual_tensors)>0
        self.nb_train_seq = len(dataset.tensor_limits_keeper.df_verif_train)


        self.dataloader = dataset.dataloader
        self.training_mode = 'train'
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.out_dim_factor = dataset.out_dim_factor
        self.step_ahead = dataset.step_ahead
        self.horizon_step = dataset.horizon_step
        self.metrics = args.metrics

        if args.loss_function_type in ['MSE','HuberLoss','masked_mae','masked_mse','huber_loss','masked_huber_loss']:
            self.type_calib = None
            self.alpha = None
            args.track_pi = False
        elif args.loss_function_type == 'quantile':
            self.type_calib = args.type_calib
            self.alpha = args.alpha
        else:
            raise NotImplementedError(f"metrics associated to {args.loss_function_type} has not been implemented")
        

        self.load_model(args)
        model_memory_cost(self.model)
        

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

    def load_model(self,args):
        if args.model_name == 'SARIMAX':
            self.model = SARIMAX(
                 order=args.order,
                 seasonal_order=args.seasonal_order,
                 enforce_stationarity=args.enforce_stationarity,
                 enforce_invertibility=args.enforce_invertibility,
                 )
        elif args.model_name == 'XGBoost':
            self.model = XGBoost(n_estimators=100,
                 learning_rate=args.lr,
                 max_depth=args.max_depth,
                 subsample=args.subsample,
                 colsample_bytree=args.colsample_bytree,
                 gamma=args.gamma,
                 reg_alpha=args.reg_alpha,
                 reg_lambda=args.reg_lambda,
                 objective=f'reg:{args.reg}',
                 eval_metric=args.eval_metric,
            )

    def train_valid_ML_model(self,normalizer):
        inputs_train = [[x,y,x_c] for x,y,x_c in self.dataloader['train']]
        X_train = torch.cat([x for x,_,_ in inputs_train])
        X_exog_train = torch.cat([x for x,_,_ in inputs_train]) # ....


        inputs_valid = [[x,y,x_c] for x,y,x_c in self.dataloader['valid']]
        X_train = torch.cat([x for x,_,_ in inputs_train])
        inputs_test = [[x,y,x_c] for x,y,x_c in self.dataloader['test']]

        self.model.forward(x,x_vision,x_calendar)

    def get_test_accuracy(self,normalizer):
        
        return normalizer


    def train_and_valid(self,normalizer = None,df_verif_test = None,station = 0,unormalize_loss = None):
        print(f'\nstart training')
        results_df = None
        self.chrono = Chronometer()
        self.chrono.start()
        max_memory = 0
        

        # Train and Valid each epoch 
        self.train_valid_ML_model(normalizer=normalizer if unormalize_loss else None)

        # Test Inferrence: 
        self.get_test_accuracy(normalizer)

        # Keep track on cpu-usage 
        max_memory = get_cpu_usage(max_memory)


        # Allow to keep track on the final metrics, and if the trial is 'terminated'
        if (not(self.args.ray)):
            self.performance = { 'test_metrics':self.performance['test_metrics'],
                                'fold': self.args.current_fold,
                                'Total_training_time': str(datetime.now()- self.chrono.start_proc),
                                'Training_perf': f"{'{:.2f}'.format(np.sum(self.chrono.time_perf_train))}s train" ,
                                'Loading_perf' : f"{'{:.2f}'.format(np.sum(self.chrono.time_perf_load))}s loading"
                                }
                                    

        self.chrono.stop()
        self.chrono.display()
        print(print_memory_usage(max_memory))
        torch.cuda.empty_cache()

        return(results_df)
    

            
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

 

