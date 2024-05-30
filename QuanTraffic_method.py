import pickle
import numpy as np 
import pandas as pd
import os 

from utilities_DL import get_MultiModel_loss_args_emb_opts,load_init_trainer
from DL_class import MultiModelTrainer

from config import get_parameters
from paths import folder_path,file_name,save_folder
import torch



def split_tensor(X,split_prop):
    return(X[:int(len(X)*split_prop)],X[int(len(X)*split_prop):])

def split_calibration_dataset(trainer,split_prop) :
    data = [[x_b,y_b,t_b[trainer.args.calendar_class]] for  x_b,y_b,*t_b in trainer.dataloader['cal']]

    X_cal = torch.cat([x_b for [x_b,_,_,_] in data]).to(trainer.args.device),
    Y_cal= torch.cat([y_b for [_,y_b,_,_] in data]).to(trainer.args.device)
    T_pred = torch.cat([t_pred for [_,_,t_pred,_] in data]).to(trainer.args.device)

    X_cal_train,X_cal_valid = split_tensor(X_cal,split_prop)
    Y_cal_train,Y_cal_valid = split_tensor(Y_cal,split_prop)
    T_pred_train,T_pred_valid = split_tensor(T_pred,split_prop)

    return(X_cal_train,X_cal_valid,Y_cal_train,Y_cal_valid,T_pred_train,T_pred_valid)

def prediction_on_specific_dataset(trainer,X,T):
    return(trainer.model(X,T))

def get_predictions(trainer,split_prop):
    X_cal_train,X_cal_valid,Y_cal_train,Y_cal_valid,T_pred_train,T_pred_valid = split_calibration_dataset(trainer,split_prop)
    Y_cal_train_pred = prediction_on_specific_dataset(trainer,X_cal_train,T_pred_train)
    Y_cal_valid_pred = prediction_on_specific_dataset(trainer,X_cal_valid,T_pred_valid)

    return(Y_cal_train_pred,Y_cal_train,Y_cal_valid_pred,Y_cal_valid)

def compute_error(Y_true,Y_pred_lower,Y_pred_upper):
    # 2 . ==== Compute error :     
    #   error_low = target - lower_band  / error_high  = upper_band - target
    #   err_dis = torch.cat([error_low,error_high])   # Concat errors on the dim 0 : [B,S,N],[B,S,N] -> [2*B,S,N]    , with S =nb step ahead of the prediction
    # ==== ....
    error_low = Y_true - Y_pred_lower
    error_high = Y_pred_upper - Y_true
    err =  torch.cat([error_low,error_high])

    return err 






if __name__ == '__main__': 

    # ===== Load Dataset, Trainer, dataloader ...
    save_model_folder = f"{save_folder}best_models/"
    model_perf_path = f"{save_model_folder}model_args.pkl"
    model_perf = pickle.load(open(model_perf_path,'rb'))
    best_valid_loss = np.inf 
    for trial in model_perf['model'].keys():

        valid_loss = model_perf['model'][trial]['performance']['valid_loss']
        loss_function  = model_perf['model'][trial]['args']['loss_function_type']
        if loss_function == 'quantile':
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_trial = trial

    best_model = f"{save_model_folder}{best_trial}_best_model.pkl"

    print(best_model)

    config = model_perf['model'][best_trial]['args']
    config['abs_path'] = f"{os.path.abspath(os.getcwd())}/"
    config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config['K_fold'] = 1
    args = get_parameters(config)

    Datasets,DataLoader_list,dic_class2rpz,nb_words_embedding,time_slots_labels,dic_rpz2class = load_init_trainer(folder_path,file_name,args)
    (loss_function,Model_list,Optimizer_list,Scheduler_list,args_embedding) = get_MultiModel_loss_args_emb_opts(args,nb_words_embedding,dic_class2rpz,n_vertex = len(Datasets[0].columns))
    multimodeltrainer = MultiModelTrainer(Datasets,Model_list,DataLoader_list,args,Optimizer_list,loss_function,Scheduler_list,args_embedding=args_embedding,dic_class2rpz=dic_class2rpz,show_figure=False)
    saved_checkpoint = torch.load(best_model)
    trainer = multimodeltrainer.Trainers[0]
    trainer.model.load_state_dict(saved_checkpoint['state_dict'])
    # ===== ....


    # ==== QuanTraffic Calibration : 
    # Load Y_pred_calibration1 and Y_pred_calibration2
    split_prop = 0.5

    Y_cal_train_pred,Y_cal_train,Y_cal_valid_pred,Y_cal_valid = get_predictions(trainer,split_prop)    # Get prediction from Calibration DataSet 
    Y_cal_train_pred_lower,Y_cal_train_pred_upper = Y_cal_train_pred[...,0],Y_cal_train_pred[...,1]    # Get Upper and Lower band from prediction
    Y_cal_valid_pred_lower,Y_cal_valid_pred_upper = Y_cal_valid_pred[...,0],Y_cal_valid_pred[...,1]

    err = compute_error(Y_cal_train,Y_cal_train_pred_lower,Y_cal_train_pred_upper)  # Get Prediction

    # ....




# 1 . ==== 'Preprocessing' 
    #   Data .to(device), DataLoader(XS,YS)
    #   Load trained model
    #   Prediction : YS_pred (quantile_l and quantile_u) on calibration dataset 
    #   Inverse_transform Prediction ( ??? )
 
    # split YS_pred in 2 part: upper and lower estimation (YS_pred_O and YS_pred1)
    #   split  YS and YS_pred in 2 part: 'train' and 'validation'  (YS_train,YS_val for the targets ones, and YS_0_train,YS_1_train and YS_0_val,YS_1_val for the predicted ones )
    #   Traffic state (demand, speed, flow) are always >= 0, then mask 'YS_1_train' which is the lower band, by setting all negative values by 0.

    #   error_quantile = 'n_grid'
    # ==== ....
    
    # 2 . ==== Compute error :     
    #   error_low = target - lower_band  / error_high  = upper_band - target
    #   err_dis = torch.cat([errow_low,error_high])   # Concat errors on the dim 0 : [B,S,N],[B,S,N] -> [2*B,S,N]    , with S =nb step ahead of the prediction
    # ==== ....

    # 3 . ==== Compute Quantile Table : 
    # Thanks to all residual error, compute a quantile for each percentile, each node id, each 't' 
    # corr_err_list = Quantile_table, shape : [error_quantile, S, N]
    # ==== ....

    # 4 . ==== PICP and MPIW Table:
    # For each quantile (percentile here, error_quantile = 99):
    #    Compute lower_band:  lower_prediction - quantile table for the quantile q    -> Shape [B,S,N]
    #    Compute upper_band:  upper_prediction + quantile table for the quantile q   -> Shape [B,S,N]  (on s'assure que lower_band > 0)
    #    Compute PICP (coverage_list) et MPIW (interval_list) mean through first dimension.  shape [S,N]
    # Compute it for each quantile  -> coverage_list and interval_list Shape [error_quantile,S,N]
    # ==== ....


    # 5 . ==== Compute PICP and MPIW MinMax_Normalization : (???) 
    #   Compute Normalized PICP  interval_nor (resp MPIW  - coverage_nor) through dim 0, For each node, each time-step ahead, 
    # ==== ....

    # 6 . ==== For each Lambda coeff, choose the best Quantile :
    #  For each lambda : 
    #    Loss = (1-i)MPIW_n  - i PICP_n    -> shape [error_quantile,S,N]
    #    find index table of min_loss for each couple (s,n) 
    #    for each s, each n : 
    #        cor_err[lambda,s,n] = Quantile_table[index[s,n],s,n]
    #
    # Return a Calibration table 'cor_err' of shape [lambda_list,S,N] 
    # ==== ....

    # 7 . ==== For each Lambda coeff, calibration on the Validation Prediction 
    #  For each lambda_i: 
    #    Compute lower band : lower_validation_predcition - calibration table[lambda_i]
    #    Compute upper band : upper_validation_predcition + calibration table[lambda_i]
    #    Compute coverage (independent_coverage), a Boolean Tensor, shape [B2,S,N]. True if real value within PI
    #    Compute PICP : mean of 'True coverage', shape [1]
    #
    # do for each lambda_i, then produce a PICP vector [lambda list]
    # Return the best lambda_i(the closest from expected quantile 0.9)
    # ==== .... 


    # 6 and 7 are the optimisation part to select the best_lambda
    # Then, for each 's' and each 'n', we have the associated best calibration cor_err[best_lambda,s,n]

    # These are the final calibration scores and can be used on test_dataset