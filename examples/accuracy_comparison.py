# GET PARAMETERS
import os 
import sys
import pandas as pd
import matplotlib.pyplot as plt 
import torch 
import pickle
import numpy as np 
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from examples.load_best_config import load_trainer_ds_from_saved_trial
from plotting.plotting import plot_coverage_matshow,get_df_mase_and_gains,get_df_gains
from examples.train_model import load_init_model_trainer_ds
import numpy as np 
import pandas as pd
import torch
from bokeh.plotting import figure, show,output_notebook
from bokeh.models import Legend,DatetimeTickFormatter
from bokeh.layouts import row,column
from calendar_class import is_bank_holidays

# Personnal imports: 
from bokeh.models import ColumnDataSource
from bokeh.palettes import Blues9,Reds9 
from bokeh.palettes import Plasma256 
from bokeh.palettes import Turbo256 as palette

##### ==================================================

##### ==================================================
def plot_daily_profile(profil_real_station_i,daily_profil1_station_i,daily_profil2_station_i,
                        station,
                        width = 1200,
                        height = 400,
                       ):
    title = f"Daily Profile Comparison for Station: {station}"
    p =  figure(title=title,width=width,height=height,x_axis_type="datetime")

    df = profil_real_station_i.copy()
    for c in df:
        p.line(x=df.index, y=df[c], alpha=0.8,color = 'green',
                legend_label=c, 
                #line_dash="dashed",
                line_width=1,
                )
        
    df = daily_profil1_station_i.copy()
    for k,c in enumerate(df):
        p.line(x=df.index, y=df[c], alpha=0.8,color = Blues9[k],
                legend_label=c, 
                #line_dash="dashed",
                line_width=1,
                )


    df = daily_profil2_station_i.copy()
    for k,c in enumerate(df):
        p.line(x=df.index, y=df[c], alpha=0.8,color = Reds9[k],
                legend_label=c, 
                #line_dash="dashed",
                line_width=1,
                )


    p.xaxis.major_label_orientation = 1.2  # Pour faire pivoter les labels des x
    # legend = Legend(items=legend_it)
    # legend.click_policy="hide"
    p.legend.click_policy = "hide"
    # p.add_layout(legend, 'right')
    p.xaxis.formatter=DatetimeTickFormatter(
        seconds="%H:%M",
        minsec="%H:%M",
        minutes="%H:%M",
        hourmin="%H:%M",
        hours="%H:%M",
        days="%H:%M",
        months="%H:%M",
        years="%H:%M"
                        )

    output_notebook()
    show(p)

def get_working_day_daily_profile_on_h(h,full_predict1,df_verif,ds1,args_init1):
        dates = df_verif.iloc[:,-(args_init1.step_ahead+1-h)].copy()
        #print('df verif at iloc: ',-(args_init1.step_ahead+1-h))
        #display(dates.head(2))
        dates.reset_index(drop=True,inplace=True)

        ## Select Only Working Days : 
        dates_is_bank_holidays = dates.apply(lambda x: is_bank_holidays(x,city= ds1.city))
        dates_is_weekday = dates.dt.dayofweek < 5
        dates_is_working_day = ~dates_is_bank_holidays & dates_is_weekday


        df_dates = pd.DataFrame({'date':dates,
                    'working_days':dates_is_working_day,
                    })
        

        df_horizon = pd.DataFrame(full_predict1[:,:,h-1].detach().cpu().numpy(),columns = ds1.spatial_unit)
        # print('Iloc: ',h-1)
        df_horizon = pd.concat([df_dates.copy(),df_horizon],axis=1)
        df_horizon['hour'] = df_horizon['date'].dt.hour
        df_horizon['minute'] = df_horizon['date'].dt.minute
        df_horizon = df_horizon[df_horizon['working_days']].copy().drop(columns = ['working_days','date'])
        df_horizon = df_horizon.groupby(['hour','minute']).mean().reset_index()

        ## build fake date from hour/minute as xticks and plot columns: 
        df_horizon['hour'] = df_horizon['hour'].apply(lambda x: f"{x:02d}")
        df_horizon['minute'] = df_horizon['minute'].apply(lambda x: f"{x:02d}")
        df_horizon['time'] = pd.to_datetime(df_horizon['hour'] + ':' + df_horizon['minute'], format='%H:%M')
        df_horizon = df_horizon.drop(columns=['hour','minute'])
        df_horizon.set_index('time', inplace=True)
        return df_horizon
        

def working_day_daily_profile(full_predict1,ds1,args_init1):
    df_verif = ds1.tensor_limits_keeper.df_verif_test.copy() 
    profil_per_horizon = {}
    for h in range(args_init1.horizon_step, args_init1.step_ahead+1,args_init1.horizon_step):
        #   print('\n\nh: ',h)
          profil_per_horizon[h] = get_working_day_daily_profile_on_h(h,
                                                                     full_predict1,
                                                                     df_verif,
                                                                     ds1,
                                                                     args_init1)
    return profil_per_horizon


def get_profil_per_horizon(full_predict1,full_predict2,Y_true,ds1,ds2,args_init1,args_init2):
    profil1_per_horizon = working_day_daily_profile(full_predict1,ds1,args_init1)
    profil2_per_horizon = working_day_daily_profile(full_predict2,ds2,args_init2)
    # profil_real = working_day_daily_profile(Y_true,ds1,args_init1)
    profil_real = get_working_day_daily_profile_on_h(args_init1.step_ahead,
                                                     Y_true,
                                                     ds1.tensor_limits_keeper.df_verif_test,
                                                     ds1,
                                                     args_init1)


    ## Process columns Name:
    profil_real.rename(columns = {col: f"Real {col}" for col in profil_real.columns}, inplace=True)
    for h in range(args_init1.horizon_step, args_init1.step_ahead+1,args_init1.horizon_step):
        profil1_per_horizon[h].rename(columns = {col: f"Model1 {col} h{h}" for col in profil1_per_horizon[h].columns}, inplace=True)
        profil2_per_horizon[h].rename(columns = {col: f"Model2 {col} h{h}" for col in profil2_per_horizon[h].columns}, inplace=True)
    return profil_real,profil1_per_horizon,profil2_per_horizon



##### ==================================================


##### ==================================================



def plot_profile_comparison_between_2_prediction(args_init1,full_predict1,full_predict2,real,ds1,station_i,station, width=900, height=400, bool_plot = True):
    title = f'Comparison of prediction of {station} between two models'
    p = figure(title=title,x_axis_type='datetime', x_axis_label='Time', y_axis_label='Demand volume', width=width, height=height)
    p.add_layout(Legend(), 'right')
    df_verif = ds1.tensor_limits_keeper.df_verif_test.copy()

    # Predicted Values
    for k,step_ahead in enumerate(range(args_init1.horizon_step, args_init1.step_ahead+1,args_init1.horizon_step)):
        p.line(df_verif.iloc[:,-(args_init1.step_ahead//args_init1.horizon_step-k)], full_predict1[:,station_i,k].cpu().numpy(), 
            legend_label=f"predict1-h{step_ahead}", 
            line_dash="dashed", line_width=1, color="green")

        p.line(df_verif.iloc[:,-(args_init1.step_ahead//args_init1.horizon_step-k)], full_predict2[:,station_i,k].cpu().numpy(), 
            legend_label=f"predict2-h{step_ahead}", 
            line_dash="dashed", line_width=1, color="red")
    # ...

    # Real Values: 
    p.line(df_verif.iloc[:,-1], real[:,station_i].cpu().numpy(), legend_label="True Value", line_width=2, color="blue")
    p.legend.click_policy = "hide"

    p.xaxis.formatter=DatetimeTickFormatter(
              months="%b",
              days="%a %d %b",
              hours="%a %d %b %H:%M",
              minutes="%a %d  %H:%M"
                     )

    if bool_plot:
        output_notebook()
        show(p)
    return p




def get_predict_real_and_inputs(trainer1,trainer2,ds1,ds2,training_mode):
    """
    Get the full predictions and real values from two trainers and datasets.
    Also get the inputs used for the predictions.
    Args:
        trainer1 (Trainer): First trainer.
        trainer2 (Trainer): Second trainer.
        ds1 (Dataset): First dataset.
        ds2 (Dataset): Second dataset.
        training_mode (str): Mode for training, e.g., 'train', 'val', 'test'.
    """ 
    full_predict1,Y_true,_ = trainer1.testing(ds1.normalizer, training_mode =training_mode)
    full_predict2,_,_ = trainer2.testing(ds2.normalizer, training_mode =training_mode)

    inputs = [[x,y,x_c] for  x,y,x_c in ds1.dataloader[training_mode]]
    X = torch.cat([x for x,_,_ in inputs],0)
    X = ds1.normalizer.unormalize_tensor(inputs = X,feature_vect = True) # unormalize input cause prediction is unormalized 
    return(full_predict1,full_predict2,Y_true,X)

def get_previous_and_prediction(full_predict1,full_predict2,Y_true,X,step_ahead,step_ahead_max):
    previous = X[:,:,-(step_ahead_max+1-step_ahead)]
    predict1 = full_predict1[:,:,step_ahead-1]
    predict2 = full_predict2[:,:,step_ahead-1]
    real = Y_true[:,:,step_ahead-1]
    return previous,predict1,predict2,real


def plot_gain_between_models_with_temporal_agg(ds,dic_error,stations,temporal_aggs,training_mode,metrics,step_ahead):
    """
    Plot the gain between two models for different temporal aggregations.
    Args:
        ds (Dataset): Dataset containing the data.
        dic_error (dict): Dictionary containing error metrics.
        stations (list): List of stations to consider.
        temporal_aggs (list): List of temporal aggregations to consider :
            >>>> choices = ['hour', 'date', 'weekday', 'weekday_hour', 'weekday_hour_minute', 'daily_period', 'working_day_hour']
        training_mode (str): Mode for training, e.g., 'train', 'val', 'test'.
        metrics (list): List of metrics to compute gains for.
        step_ahead (int): Step ahead for predictions.
    Returns:
        dic_gain_agg (dict): Dictionary containing gain values for each metric and temporal aggregation.
        heatmap plot for each metric and temporal aggregation.
    """

    dic_gain_agg = {metric : {} for metric in metrics}
    dic_error_agg  = {metric : {} for metric in metrics}
    for metric in metrics:
        if len(temporal_aggs) == 1:
            fig, axes = plt.subplots(1, 2, figsize=(max(8,0.5*len(stations)),6))
        else:
            if temporal_aggs == ['hour','date','weekday']:
                gridspec_kw={'width_ratios': [1,5],'height_ratios': [4,3,2]}
            else:
                gridspec_kw={'width_ratios': [1,5],'height_ratios': [1]*len(temporal_aggs)}
            
            fig, axes = plt.subplots(len(temporal_aggs), 2, figsize=(max(8,0.5*len(stations)),6*len(temporal_aggs)),gridspec_kw=gridspec_kw)
    
        for i,temporal_agg in enumerate(temporal_aggs):
            if metric == 'mase':
                df_gain21,error_pred1_agg,error_pred2_agg = get_df_mase_and_gains(ds,dic_error,training_mode,temporal_agg,stations)
            else:
                df_gain21,error_pred1_agg,error_pred2_agg = get_df_gains(ds,dic_error,metric,training_mode,temporal_agg,stations)
            ### --- AGG All sations  
            if len(temporal_aggs) == 1:
                plt.sca(axes[0])
            else:
                plt.sca(axes[i,0])
            plot_coverage_matshow(pd.DataFrame(pd.DataFrame(df_gain21).mean(axis=1)),cmap = 'RdYlBu', save=None, 
                                cbar_label=f'{metric.upper()} Gain (%)',bool_reversed=True,v_min=-10,v_max=10,display_values = True)
            title = f"Average {metric.upper()} Gain(%) per {temporal_agg} of \nModel2 compared to Model1 at horizon {step_ahead}"
           
            if len(temporal_aggs) == 1:
                axes[0].set_title(f"{title}\nAggregated through stations")
            else:
                axes[i,0].set_title(f"{title}\nAggregated through stations")
            ## ...

            ### --- Per station 
            if len(temporal_aggs) == 1:
                plt.sca(axes[1])
            else:
                plt.sca(axes[i,1])
            plot_coverage_matshow(pd.DataFrame(df_gain21),cmap = 'RdYlBu', save=None, 
                                cbar_label=f'{metric.upper()} Gain (%)',bool_reversed=True,v_min=-20,v_max=20,display_values = False)
            if len(temporal_aggs) == 1:
                axes[1].set_title(title)
            else:
                axes[i,1].set_title(title) 
            dic_gain_agg[metric][temporal_agg] = df_gain21
            dic_error_agg[metric][temporal_agg] = {'error_pred1_agg': error_pred1_agg, 'error_pred2_agg': error_pred2_agg}

            ## ...
    return dic_gain_agg,dic_error_agg

def load_trainer_ds_from_2_trials(trial_id1,trial_id2,modification,model_args,path_model_args):
    """
    Load trainer and dataset from two trials.
    Will be used to compare the two models.
    """
    args = model_args['model'][trial_id1]['args']
    model_save_path = f"{path_model_args}/{trial_id1}.pkl"
    trainer1, ds1, args_init1 = load_trainer_ds_from_saved_trial(args,model_save_path,modification = modification)


    args = model_args['model'][trial_id2]['args']
    model_save_path = f"{path_model_args}/{trial_id2}.pkl"
    trainer2, ds2, args_init2 = load_trainer_ds_from_saved_trial(args,model_save_path,modification = modification)
    return trainer1,trainer2,ds1,ds2,args_init1,args_init2 


def load_trainer_ds_from_1_args(args_init,modification = {},save_folder = None,trial_id = None,fold_to_evaluate=None):
    if fold_to_evaluate is None:
        fold_to_evaluate=[args_init.K_fold-1]
    trainer,ds,model,args = load_init_model_trainer_ds(fold_to_evaluate,save_folder,args_init,modification,trial_id)  
    return trainer,ds 


def print_global_info(trial_id1,trial_id2,full_predict1,full_predict2,Y_true,ds1):
    print('Model1 correspond to : ',trial_id1)
    print('Model2 correspond to : ',trial_id2)
    horizon_list = f"{'/'.join(list(map(str,np.arange(1,ds1.step_ahead+1))))}"
    for metric in ['mae','mse','rmse']:
        for horizon_group in ['per','all']:
            if horizon_group == 'all': axis = None
            if horizon_group == 'per': axis = [0,1]

            if metric == 'mae':
                err1 = (Y_true - full_predict1).abs().mean(axis =axis)
                err2 = (Y_true - full_predict2).abs().mean(axis = axis)
            elif metric == 'mse':
                err1 = ((Y_true - full_predict1)**2).mean(axis = axis)
                err2 = ((Y_true - full_predict2)**2).mean(axis = axis)
            elif metric == 'rmse':
                err1 = ((Y_true - full_predict1)**2).mean(axis = axis).sqrt()
                err2 =  ((Y_true - full_predict2)**2).mean(axis = axis).sqrt()

            if horizon_group == 'per':
                gain_per_horizon = ((err2/err1 - 1)*100).numpy()
                gain_per_horizon = [f'{val:.2f}' for val in gain_per_horizon]
                gain_per_horizon = ' / '.join(gain_per_horizon)    
            if horizon_group == 'all':
                gain_all_horizon = ((err2.mean()/err1.mean() - 1)*100).numpy().item()
                gain_all_horizon = f'{gain_all_horizon:.2f}'
        print(f'Global {metric.upper()} gain (%) from Model2 compared to Model1 at horizon {horizon_list}: {gain_per_horizon} // All horizon : {gain_all_horizon}')

