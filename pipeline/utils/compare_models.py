# GET PARAMETERS
import os 
import sys
import pandas as pd
import matplotlib.pyplot as plt 
import torch 
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from examples.train_and_visu_non_recurrent import get_ds
from constants.paths import SAVE_DIRECTORY
from pipeline.high_level_DL_method import load_optimizer_and_scheduler
from pipeline.dl_models.full_model import full_model

from examples.load_best_config import load_args_of_a_specific_trial
from pipeline.trainer import Trainer
from pipeline.plotting.plotting import get_gain_from_mod1,plot_coverage_matshow,get_df_mase_and_gains
from pipeline.plotting.TS_analysis import plot_TS


def display_consistency(trainer,ds,save_folder,trial_id,add_name_id,training_mode = 'test'):
    full_predict1,Y_true,_ = trainer.testing(ds.normalizer, training_mode =training_mode)
    Y_true= Y_true.detach().clone().reshape(-1)    
    full_predict1= full_predict1.detach().clone().reshape(-1)    
    error_pred1 = ((Y_true - full_predict1)**2).mean()

    df_metrics1 = pd.read_csv(f"{SAVE_DIRECTORY}/{save_folder}/METRICS_{trial_id}{add_name_id}.csv")
    print(f"MSE errror on {training_mode} by loading trained model : {'{:.2f}'.format(error_pred1.item())}\n")
    return df_metrics1

def get_previous_and_prediction(trainer1,trainer2,ds1,ds2,training_mode):
    full_predict1,Y_true,_ = trainer1.testing(ds1.normalizer, training_mode =training_mode)
    full_predict2,_,_ = trainer2.testing(ds2.normalizer, training_mode =training_mode)

    inputs = [[x,y,x_c] for  x,y,x_c in ds1.dataloader[training_mode]]
    X = torch.cat([x for x,_,_ in inputs],0)
    X = ds1.normalizer.unormalize_tensor(inputs = X,feature_vect = True) # unormalize input cause prediction is unormalized 

    previous = X[:,:,-1]
    predict1 = full_predict1[:,:,0]
    predict2 = full_predict2[:,:,0]
    real = Y_true[:,:,0]
    return previous,predict1,predict2,real

def plot_gain_between_models_with_temporal_agg(ds,dic_error,stations,temporal_aggs,training_mode):
    fig, axes = plt.subplots(len(temporal_aggs), 2, figsize=(max(8,0.5*len(stations)),6*len(temporal_aggs)),gridspec_kw={'width_ratios': [1,5],'height_ratios': [4,3,2]})
    for i,temporal_agg in enumerate(temporal_aggs):
        df_mase1,df_mase2,df_gain21 = get_df_mase_and_gains(ds,dic_error,training_mode,temporal_agg,stations)
        # agg All sations  
        plt.sca(axes[i,0])
        plot_coverage_matshow(pd.DataFrame(pd.DataFrame(df_gain21).mean(axis=1)),cmap = 'RdYlBu', save=None, 
                            cbar_label='MASE Gain (%)',bool_reversed=True,v_min=-10,v_max=10)
        title = f'Average MASE Gain(%) per {temporal_agg} of \nModel2 compared to Model1\nAggregated through stations'
        axes[i,0].set_title(title)

        # Per station 
        plt.sca(axes[i,1])
        plot_coverage_matshow(pd.DataFrame(df_gain21),cmap = 'RdYlBu', save=None, 
                            cbar_label='MASE Gain (%)',bool_reversed=True,v_min=-20,v_max=20)
        title = f'Average MASE Gain(%) per {temporal_agg} of \nModel2 compared to Model1'
        axes[i,1].set_title(title) 

def get_trainer_and_ds_from_saved_trial(trial_id,add_name_id,save_folder,modification,fold_to_evaluate = None):
    # Load Data and Init Model:
    if fold_to_evaluate is None:
        fold_name = 'complete_dataset'
    else:
        fold_name = fold_to_evaluate[0]

    #args,_ = load_configuration(trial_id1,load_config=True)
    args = load_args_of_a_specific_trial(trial_id,add_name_id,save_folder,fold_name)

    if fold_to_evaluate is None:  fold_to_evaluate = [args.K_fold-1]

    
           
    ds,_,_,_,_ =  get_ds(args_init=args,modification = modification,fold_to_evaluate=fold_to_evaluate)
    model = full_model(ds, args).to(args.device)


    # Load Trained Weights 
    model_param = torch.load(f"{SAVE_DIRECTORY}/{save_folder}/best_models/{trial_id}{add_name_id}_f{fold_name}.pkl")
    model.load_state_dict(model_param['state_dict'],strict=True)


    # Load Trainer : 
    optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
    trainer = Trainer(ds,model,args,optimizer,loss_function,scheduler = scheduler)

    return trainer,ds,args