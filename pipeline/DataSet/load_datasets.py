
# GET PARAMETERS
import sys
import os
import pandas as pd
import numpy as np 
import torch
if torch.cuda.is_available():
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32  = True
   
from argparse import Namespace
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from pipeline.K_fold_validation.K_fold_validation import KFoldSplitter
from pipeline.utils.save_results import get_trial_id
from constants.config import modification_contextual_args,update_modif, local_get_args



def get_ds_without_shuffling_on_train_set(trainer,modification,args_init, fold_to_evaluate):
    modification.update({'shuffle':False,
                         'data_augmentation':False })
    ds_no_shuffle,_,_,_,_ =  get_ds(modification = modification,
                                        args_init=args_init,
                                        fold_to_evaluate=fold_to_evaluate)
    trainer.dataloader = ds_no_shuffle.dataloader
    return trainer,ds_no_shuffle



def get_ds(model_name=None,dataset_names=None,dataset_for_coverage=None,
           modification = {},
           args_init = None, 
           fold_to_evaluate = None
            ):
    args_with_contextual,K_subway_ds = get_multi_ds(model_name if model_name is not None else args_init.model_name,
                                                    dataset_names if dataset_names is not None else args_init.dataset_names,
                                                    dataset_for_coverage if dataset_for_coverage is not None else args_init.dataset_for_coverage,
                                                    modification=modification,
                                                    args_init=args_init,
                                                    fold_to_evaluate=fold_to_evaluate)
    ds = K_subway_ds[-1]
    trial_id = get_trial_id(args_with_contextual)
    save_folder = None
    df_loss= pd.DataFrame()

    return(ds,args_with_contextual,trial_id,save_folder,df_loss)

def get_inputs(args,folds):
    K_fold_splitter = KFoldSplitter(args,folds)
    K_subway_ds,args = K_fold_splitter.split_k_fold()

    ## Specific case if we want to validate on the init entiere dataset:
    if (args.evaluate_complete_ds and args.validation_split_method == 'custom_blocked_cv'): 
        subway_ds,_,_ = K_fold_splitter.load_init_ds(normalize = True)
        K_subway_ds.append(subway_ds)

    
    return(K_fold_splitter,K_subway_ds,args)


def get_multi_ds(model_name,
                 dataset_names,
                 dataset_for_coverage,
                 modification = {},
                 args_init = None, 
                 fold_to_evaluate = None):

    # Tricky but here we need to set 'netmob' so that we will use the same period for every combination
    if args_init is None:
        args_copy = local_get_args(model_name,
                                    args_init=None,
                                    dataset_names=dataset_names,
                                    dataset_for_coverage=dataset_for_coverage,
                                    modification = modification)
    else:
        args_copy = Namespace(**vars(args_init))
        for key,values in modification.items():
            setattr(args_copy,key,values)
        args_copy = update_modif(args_copy)

    # Add [0] in folds according the presence of 'hp_tuning_on_first_fold' or not : 
    # If we didn't precise, we try to evaluate the fold [0] (the shorter one)
    if fold_to_evaluate is None:
        if (args_copy.K_fold > 1) and (args_init.hp_tuning_on_first_fold):
            folds = [1] # [0,1]
        else:
            folds = [0] # [0,0]

    # If we precise wich fold, we have to add fold [0] if args_init.hp_tuning_on_first_fold is set to True
    else:
        folds = fold_to_evaluate
        #if args_init.hp_tuning_on_first_fold:
        #    folds = [0] + folds

        
    K_fold_splitter,K_subway_ds,args_with_contextual = get_inputs(args_copy,folds)

    # Weird tricks cause folds can be np array or list 
    target = np.arange(args_init.K_fold) 
    comparison = folds == target if isinstance(folds, np.ndarray) else folds == list(target)
    condition_i = comparison.all() if isinstance(comparison, np.ndarray) else bool(comparison)
    if (args_init.hp_tuning_on_first_fold) & (condition_i):
        K_subway_ds = K_subway_ds[1:]

    args_with_contextual = modification_contextual_args(args_with_contextual,modification)


    save_reproductible = False
    if save_reproductible:
        ds_i = K_subway_ds[-1]
        path_save = os.path.expanduser('~/prediction-validation/save/data')
        np.save(open(f'{path_save}/U_train_scaled.npy','wb'), ds_i.U_train.detach().cpu().numpy())
        np.save(open(f'{path_save}/Utarget_train_scaled.npy','wb'), ds_i.Utarget_train.detach().cpu().numpy())
        np.save(open(f'{path_save}/U_valid_scaled.npy','wb'), ds_i.U_valid.detach().cpu().numpy())
        np.save(open(f'{path_save}/Utarget_valid_scaled.npy','wb'), ds_i.Utarget_valid.detach().cpu().numpy())
        np.save(open(f'{path_save}/U_test_scaled.npy','wb'), ds_i.U_test.detach().cpu().numpy())
        np.save(open(f'{path_save}/Utarget_test_scaled.npy','wb'), ds_i.Utarget_test.detach().cpu().numpy())

        for contextual_name,contextual_tensors in ds_i.contextual_tensors.items():
            for training_mode,contextual_tensor in contextual_tensors.items():
                np.save(open(f'{path_save}/{contextual_name}_{training_mode}.npy','wb'), contextual_tensor.detach().cpu().numpy())


    return args_with_contextual,K_subway_ds
