import os 
import sys
import torch 
import torch._dynamo as dynamo; dynamo.graph_break()
torch._dynamo.config.verbose=True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..','..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from examples.HP_parameter_choice import hyperparameter_tuning
from examples.train_model_on_k_fold_validation import train_model_on_k_fold_validation,load_configuration
from examples.benchmark import local_get_args

def HP_and_valid_one_config(args,epochs_validation,num_samples):
    # HP Tuning on the first fold
    analysis,trial_id = hyperparameter_tuning(args,num_samples)

    # K-fold validation with best config: 
    modification = {'epochs':epochs_validation,
                    'expanding_train': None,
                    'graph_subset': None,
                    }
    train_model_on_k_fold_validation(trial_id,load_config=True,save_folder='K_fold_validation/training_with_HP_tuning',modification=modification)
    return trial_id

if __name__ == '__main__':
    model_name = 'STAEformer'
    target_data = 'PeMS08_flow'
    epochs = 200 # 200
    dataset_for_coverage = [target_data] 
    dataset_names = [target_data,'calendar']
    args = local_get_args(model_name,
                          args_init = None,
                          dataset_names=dataset_names,
                          dataset_for_coverage=dataset_for_coverage,
                          modification = {'target_data' :target_data,
                                        'ray':True,

                                        # Expanding Train & Graph Subset: 
                                        'expanding_train': 0.1,
                                        'graph_subset': 0.1,
                                        'batch_size': 128, # 16
                                         # ----

                                        'grace_period':20,
                                        'HP_max_epochs':epochs, #1000, #300,
                                        'epochs':epochs,
                                        'K_fold': 2,

                                        'evaluate_complete_ds' : True,
                                        'calendar_types':['dayofweek', 'timeofday'],
                                       'dataset_for_coverage': ['PeMS08'],
                                        'use_target_as_context': False,
                                        'data_augmentation': False,
                                        'step_ahead': 12,
                                        'station' : [],
                                        'freq': '5min',
                                        'H':12,
                                        'D':0,
                                        'W':0,

                                        'input_embedding_dim': 24,
                                        'tod_embedding_dim': 24,
                                        'dow_embedding_dim': 24,
                                        'spatial_embedding_dim':0,
                                        'adaptive_embedding_dim': 80,

                                        'node_dim': 64,
                                        'feed_forward_dim': 256,
                                        'num_heads': 4,
                                        'num_layers': 3,

                                        'use_mixed_proj': True,


                                        'optimizer': 'adamw',
                                        'lr': 0.001, # 0.001
                                        'weight_decay': 0.0015,
                                        'torch_scheduler_type': 'MultiStepLR',
                                        'loss_function_type':'HuberLoss',
                                        'torch_scheduler_milestone': [25, 45, 65],
                                        'torch_scheduler_gamma':0.1,
                                        'train_prop': 0.6,
                                        'valid_prop': 0.2,
                                        'test_prop': 0.2,
                                        'dropout':0.1,
                                        'standardize': True,
                                        'minmaxnorm': False,
                                        'metrics':['masked_mae','masked_rmse','masked_mape','masked_mse','mae','rmse','mape','mse','mase'],
                                        'unormalize_loss' : True,

                                        'num_workers' : 0, #4, # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
                                        'persistent_workers' : False ,# True 
                                        'pin_memory' : False ,# True 
                                        'prefetch_factor' : None, # 4, # None, 2,3,4,5 ... 
                                        'drop_last' : False,  # True
                                        'mixed_precision' : False, # True # False
                                        'torch_compile' : False,# 'compile', # 'compile' # 'jit_script' #'trace'
                                            }
                        )


    if True:
        epochs_validation = epochs #1000
        num_samples = 200 # 200#200
        HP_and_valid_one_config(args,epochs_validation,num_samples)

    # Minimal Test: 
    if False: 
        from pipeline.K_fold_validation.K_fold_validation import KFoldSplitter
        from pipeline.high_level_DL_method import load_optimizer_and_scheduler
        from pipeline.dl_models.full_model import full_model
        from pipeline.trainer import Trainer
        folds = [0] # Here we use the first fold for HP-tuning. In case we need to compute the Sliding K-fold validation: folds = np.arange(1,args.K_fold)

        # Split in K-fold : 
        K_fold_splitter = KFoldSplitter(args,folds)
        K_subway_ds,_ = K_fold_splitter.split_k_fold()

        # Train on the first fold: 
        subway_ds = K_subway_ds[0]
        model = full_model(subway_ds, args).to(args.device)
        optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)

        #model_ref = ray.put(model)
        trainer = Trainer(subway_ds,model,
                        args,optimizer,loss_function,scheduler = scheduler,
                        #show_figure = False,trial_id = trial_id, fold=0,save_folder = save_folder
                        )
        trainer.train_and_valid()  # No plotting, No testing, No unnormalization
