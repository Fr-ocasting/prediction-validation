import pickle
import numpy as np 
import pandas as pd
import os 

from utils.utilities_DL import get_MultiModel_loss_args_emb_opts,load_init_trainer
from trainer import MultiModelTrainer

from constants.config import convert_into_parameters
from constants.paths import FOLDER_PATH,FILE_NAME,save_folder
import torch


def split_tensor(X,split_prop):
    return(X[:int(len(X)*split_prop)],X[int(len(X)*split_prop):])

def split_calibration_dataset(trainer,split_prop) :
    data = [[x_b,y_b,t_b[trainer.args.calendar_class]] for  x_b,y_b,*t_b in trainer.dataloader['cal']]

    X_cal = torch.cat([x_b for [x_b,_,_] in data]).to(trainer.args.device)
    Y_cal= torch.cat([y_b for [_,y_b,_] in data]).to(trainer.args.device)
    T_pred = torch.cat([t_pred for [_,_,t_pred] in data]).to(trainer.args.device)

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
    error_low = Y_true.cpu().detach() - Y_pred_lower.cpu().detach()
    error_high = Y_pred_upper.cpu().detach() - Y_true.cpu().detach()
    err =  torch.cat([error_low,error_high])

    return err 


def repeat_permute(X,n_repeat):
    X = X.cpu().detach()
    return(X.repeat(n_repeat,1,1,1).permute(1,0,2,3))


def compute_quantile_of_residual_err_s_n(err,q,n,s):
    quantile_s_n = torch.quantile(err[:,n,s],q)
    return(quantile_s_n)

def compute_quantile_table_for_each_q(err,S,N,nb_quantiles = 99):
    quantile_table = torch.stack([torch.stack([torch.stack([compute_quantile_of_residual_err_s_n(err,q/(nb_quantiles+1),n,s) for s in range(S)]) for n in range(N)]) for q in range(nb_quantiles+1)])
    return(quantile_table)

def get_calibrated_metrics(Y_true,Y_pred_lower,Y_pred_upper,quantile_table,nb_quantiles):
    # Compute calibrated PI band 
    calibrated_lower_band = repeat_permute(Y_pred_lower,nb_quantiles+1) - quantile_table   
    calibrated_upper_band = repeat_permute(Y_pred_upper,nb_quantiles+1) + quantile_table  

    # Set lower band Min = 0
    mask  = calibrated_lower_band > 0  
    calibrated_lower_band = calibrated_lower_band*mask

    # Coverage table : True if real value within interval, else False
    stacked_Y_true = repeat_permute(Y_true,nb_quantiles+1)
    coverage_table = torch.logical_and(calibrated_upper_band >= stacked_Y_true,calibrated_lower_band <= stacked_Y_true)

    # Compute PICP and MPIW : 
    calibrated_PICP_table = torch.sum(coverage_table,dim = 0)/coverage_table.size(0)
    calibrated_MPIW_table = torch.mean(calibrated_upper_band - calibrated_lower_band,dim=0)

    return(calibrated_PICP_table,calibrated_MPIW_table)


def normalize_metric_table(metric_table):
    normalized_table = (metric_table - torch.min(metric_table,dim = 0)[0]) / (torch.max(metric_table,dim = 0)[0] - torch.min(metric_table,dim = 0)[0])
    return(normalized_table)

def lambda_optimization(quantile_table,PICP_table, MPIW_table,lambda_list):
    N,S = quantile_table.size(1),quantile_table.size(2)
    Loss = [(1-lambda_i)*MPIW_table - lambda_i*PICP_table for lambda_i in lambda_list]
    Index = [torch.argmin(loss,dim = 0) for loss in Loss]  # loss [nb_quantiles+1, N, S]  /  Index: list of lambda_list elmt with shape [N,S]

    best_quantile_by_lambda = []
    for i in range(len(lambda_list)):
        index_i = Index[i]  # shape [N,1]   / [N,S]
        best_quantile = torch.stack([torch.stack([quantile_table[index_i[n,s],n,s] for s in range(S)]) for n in range(N)])    # index_i[n,s] choose the 'best quantile order' (???)
        best_quantile_by_lambda.append(best_quantile)
    best_quantile_by_lambda = torch.stack(best_quantile_by_lambda)
    return(best_quantile_by_lambda)


def get_QuanTraffic_calibration_table(trainer,split_prop,nb_quantiles,lambda_list):
    # Init, Load Data:
    N = len(trainer.dataset.columns)  #40
    S = trainer.args.step_ahead #1
    Y_cal_train_pred,Y_cal_train,Y_cal_valid_pred,Y_cal_valid = get_predictions(trainer,split_prop)    # Get prediction from Calibration DataSet 
    Y_cal_train_pred_lower,Y_cal_train_pred_upper = Y_cal_train_pred[...,0].unsqueeze(-1),Y_cal_train_pred[...,1].unsqueeze(-1)    # Get Upper and Lower band from prediction
    Y_cal_valid_pred_lower,Y_cal_valid_pred_upper = Y_cal_valid_pred[...,0].unsqueeze(-1),Y_cal_valid_pred[...,1].unsqueeze(-1)

    # Tensor of Residual Error (Concat Lower band error and Upper band error)
    err = compute_error(Y_cal_train,Y_cal_train_pred_lower,Y_cal_train_pred_upper)  # Get Prediction  [2*B, N, 1] 

    # Compute Quantile Table
    quantile_table = compute_quantile_table_for_each_q(err,S,N,nb_quantiles)

    # Compute claibrated PICP and MPIW through all quantile 
    calibrated_PICP_table,calibrated_MPIW_table = get_calibrated_metrics(Y_cal_train,Y_cal_train_pred_lower,Y_cal_train_pred_upper,quantile_table,nb_quantiles)

    # Normalize them 
    n_calibrated_PICP_table,n_calibrated_MPIW_table = normalize_metric_table(calibrated_PICP_table),normalize_metric_table(calibrated_MPIW_table)

    # Choose best calibration thanks to Optimization Function 
    best_quantile_by_lambda = lambda_optimization(quantile_table,n_calibrated_PICP_table, n_calibrated_MPIW_table,lambda_list)

    # Calibration on Cal_Valid_Dataset:
    calibrated_Valid_PICP_table,calibrated_Valid_MPIW_table = get_calibrated_metrics(Y_cal_valid,Y_cal_valid_pred_lower,Y_cal_valid_pred_upper,best_quantile_by_lambda,len(lambda_list) -1)

    # Choose best Calibration argument through lambda_list different propositions: 
    best_lambda_ind = torch.argmin(torch.abs(torch.mean(calibrated_Valid_PICP_table,dim=1) - (1-trainer.args.alpha)))

    # Select Best Calibration Table : 
    final_calibration_table = best_quantile_by_lambda[best_lambda_ind]  # Simplement Q .... Semble être le même pour tout time-slot. 

    return(final_calibration_table)


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
    args = convert_into_parameters(config)

    Datasets,DataLoader_list,dic_class2rpz,nb_words_embedding,time_slots_labels,dic_rpz2class = load_init_trainer(FOLDER_PATH,FILE_NAME,args)
    (loss_function,Model_list,Optimizer_list,Scheduler_list,args_embedding) = get_MultiModel_loss_args_emb_opts(args,nb_words_embedding,dic_class2rpz,n_vertex = len(Datasets[0].spatial_unit))
    multimodeltrainer = MultiModelTrainer(Datasets,Model_list,DataLoader_list,args,Optimizer_list,loss_function,Scheduler_list,args_embedding=args_embedding,dic_class2rpz=dic_class2rpz,show_figure=False)
    saved_checkpoint = torch.load(best_model)
    trainer = multimodeltrainer.Trainers[0]
    trainer.model.load_state_dict(saved_checkpoint['state_dict'])
    # ===== ....


    # ==== QuanTraffic Calibration : 
    split_prop = 0.5
    nb_quantiles = 99
    n_lambda = 50 #41 
    lambda_list = np.arange(0,n_lambda)/n_lambda

    final_calibration_table = get_QuanTraffic_calibration_table(trainer,split_prop,nb_quantiles,lambda_list)
    # ==== ....