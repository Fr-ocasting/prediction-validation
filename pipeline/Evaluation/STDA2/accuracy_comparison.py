# GET PARAMETERS
import os 
import sys
import pandas as pd
import matplotlib.pyplot as plt 
import torch 
import pickle
import numpy as np 
import re
from bokeh.models import ColumnDataSource
from bokeh.palettes import Blues9,Reds9, Greens9
from bokeh.palettes import Plasma256 
from bokeh.palettes import Turbo256 as palette
import itertools
from pipeline.utils.metrics import load_fun
import datetime

current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Personnal Import 
from pipeline.utils.calendar_class import get_temporal_mask
from pipeline.Subset_HPO.load_best_config import load_trainer_ds_from_saved_trial
from pipeline.plotting.plotting import plot_coverage_matshow,get_df_mase_and_gains,get_df_gains,get_gain_from_mod1
from pipeline.high_level_DL_method import load_init_model_trainer_ds
from pipeline.utils.utilities import get_topk
import numpy as np 
import pandas as pd
import torch
from bokeh.plotting import figure, show,output_notebook
from bokeh.models import Legend,DatetimeTickFormatter
from bokeh.layouts import row,column
from pipeline.utils.calendar_class import is_bank_holidays 
from load_inputs.Lyon.weather import load_preprocessed_weather_df
from load_inputs.Lyon.weather import START as START_weather
from load_inputs.Lyon.weather import END as END_weather
from constants.paths import SAVE_DIRECTORY, FOLDER_PATH
from pipeline.Evaluation.STDA2.clustering.clustering import TimeSeriesClusterer
from pipeline.utils.metrics import evaluate_metrics
##### ==================================================

##### ==================================================
def plot_daily_profile(profil_real_station_i,std_profil_station_i,daily_profil1_station_i,daily_profil2_station_i,
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
        
    ## Add transparent area for std: 
    y1,y2 = df[c] - std_profil_station_i, df[c] + std_profil_station_i, 
    p.varea(x=df.index, y1=y1, y2=y2, alpha=0.2,color = Greens9[k],
            legend_label=c, 
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

def get_working_day_daily_profile_on_h(h,full_predict1,df_verif,ds1,args_init1,std = False, coeff_std = 1.0):
        h_idx = h//args_init1.horizon_step -1
        h_idx_max = args_init1.step_ahead//args_init1.horizon_step

        dates = df_verif.iloc[:,-(h_idx_max-h_idx)].copy()
        # print('\n')
        # print(' Prediction horizon: ',h)
        # print(' Prediction horizon max: ',h_idx_max)
        # print('df_verif:')
        # display(df_verif.head(2))
        # print(' Dates shape: ',dates.head(2))
        dates.reset_index(drop=True,inplace=True)

        ## Select Only Working Days : 
        dates_is_bank_holidays = dates.apply(lambda x: is_bank_holidays(x,city= ds1.city))
        dates_is_weekday = dates.dt.dayofweek < 5
        dates_is_working_day = ~dates_is_bank_holidays & dates_is_weekday


        df_dates = pd.DataFrame({'date':dates,
                    'working_days':dates_is_working_day,
                    })
        

        df_horizon = pd.DataFrame(full_predict1[:,:,h_idx].detach().cpu().numpy(),columns = ds1.spatial_unit)
        # print('Iloc: ',h-1)
        df_horizon = pd.concat([df_dates.copy(),df_horizon],axis=1)
        df_horizon['hour'] = df_horizon['date'].dt.hour
        df_horizon['minute'] = df_horizon['date'].dt.minute
        df_horizon = df_horizon[df_horizon['working_days']].copy().drop(columns = ['working_days','date'])


        if std :
            df_horizon = df_horizon.groupby(['hour','minute']).std()
            df_horizon = df_horizon * coeff_std
            df_horizon = df_horizon.reset_index()
        else:
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

    full_predict1 = full_predict1.mean(-1)
    full_predict2 = full_predict2.mean(-1)

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
    try: 
        inputs = [[x,y,x_c] for  x,y,x_c in ds1.dataloader[training_mode]]
        X = torch.cat([x for x,_,_ in inputs],0)
        # if 'weather' in trainer2.contextual_positions.keys():
        #     pos_weather = trainer2.contextual_positions['weather']
        #     X_weather = torch.cat([x_c[pos_weather] for _,_,x_c in inputs],0)
        # else:
        #     X_weather = None
    except:
        inputs = [[x,y] for  x,y in ds1.dataloader[training_mode]]
        X = torch.cat([x for x,_ in inputs],0)

        
    X = ds1.normalizer.unormalize_tensor(inputs = X,feature_vect = True) # unormalize input cause prediction is unormalized 
    return(full_predict1,full_predict2,Y_true,X)


def get_previous(X,Y_true,h_idx):
    if h_idx-2 >= 0:
        previous = Y_true[...,h_idx-2]
    else:
        previous = X[...,-1]
    return previous

def get_previous_and_prediction(full_predict1,full_predict2,Y_true,X,h_idx):
    predict1 = full_predict1[...,h_idx-1,:]
    predict2 = full_predict2[...,h_idx-1,:]
    real = Y_true[:,:,h_idx-1]
    previous = get_previous(X,Y_true,h_idx)
  
    return previous,predict1,predict2,real



def display_information_related_to_comparison(dic_error_agg_h,args_init1,metric_list,spatial_unit,step_ahead_max):
    h0 = list(dic_error_agg_h.keys())[0]
    metric0 = list(dic_error_agg_h[h0].keys())[0]
    if 'daily_period' in dic_error_agg_h[h0][metric0].keys():
        for L_metric in [[metric] for metric in metric_list]:
            print(f'\nModel: {args_init1.model_name}')
            for daily_period in ['morning_peak','evening_peak','all_day']:
                print(daily_period)
                for metric in L_metric: 
                    print(' ',metric.upper())
                    if metric == 'rmse':
                        metric_i = 'mse'
                    else:
                        metric_i = metric
                    error1_per_h = [np.mean([dic_error_agg_h[h][metric_i]['daily_period']['error_pred1_agg'][station][daily_period] for station in spatial_unit]) for h in range(args_init1.horizon_step,step_ahead_max+1,args_init1.horizon_step)]
                    error2_per_h = [np.mean([dic_error_agg_h[h][metric_i]['daily_period']['error_pred2_agg'][station][daily_period] for station in spatial_unit]) for h in range(args_init1.horizon_step,step_ahead_max+1,args_init1.horizon_step)]
                    if metric == 'rmse':
                        error1_per_h = [np.sqrt(x) for x in error1_per_h]
                        error2_per_h = [np.sqrt(x) for x in error2_per_h]

                    print('   Model 1: ',error1_per_h)
                    print('   Model 2: ',error2_per_h)


def comparison_plotting(dic_error_agg_h,full_predict1,full_predict2,ds1,Y_true,X,temporal_aggs,step_ahead,h_idx,
                        stations,training_mode,metric_list,clustered_stations,
                        folder_path =None,
                        save_name = None,
                        bool_plot = True,
                        dates = None,
                        width_ratios  = [1,10,2],
                        fig_size_x = 10,
                        fig_size_y = 6,
                        size_colorbar = "4%",
                        min_flow = 20, 
                        ):
    # Get previous and predictions
    previous,predict1,predict2,real = get_previous_and_prediction(full_predict1,full_predict2,Y_true,X,h_idx)


    # Get mean of Error Metrics (MAE, MSE, RMSE, MASE, ...): 
    L_dic_error = []
    for k in range(predict1.size(-1)):
        _,dic_error = get_gain_from_mod1(real,
                                         predict1[...,k],
                                         predict2[...,k],
                                         previous,
                                         min_flow= min_flow,
                                         metrics = ['mse','mae','mape'],
                                         acceptable_error= 0,
                                         mape_acceptable_error=0)
        L_dic_error.append(dic_error)

    agg_dic_error = {}
    for metric in  L_dic_error[0].keys():
        agg_dic_error[metric] = {}
        for k in L_dic_error[0][metric].keys():
            agg_dic_error[metric][k] = torch.stack([dic_error[metric][k] for dic_error in L_dic_error],dim=0).mean(0)
    # --- 

    # Plotting
    comparisonplotter = ComparisonPlotter(clustered_stations=clustered_stations,
                                            width_ratios= width_ratios, 
                                            fig_size_x = fig_size_x,
                                            fig_size_y = fig_size_y,
                                            size_colorbar = size_colorbar,
                                          )
    comparisonplotter.plot_gain_between_models_with_temporal_agg(
                                        ds1,dic_error,
                                        training_mode,stations, 
                                        dates,temporal_aggs,
                                        metric_list,
                                        bool_plot=bool_plot,
                                        folder_path = folder_path,
                                        save_name = save_name
                                         )
    dic_error_agg_h[step_ahead] = comparisonplotter.dic_error_agg

    return dic_error_agg_h,real



def plot_analysis_comparison_2_config(trial_id1,
                                      trial_id2,
                                      full_predict1,
                                      full_predict2,
                                      Y_true,
                                      X,
                                      ds1,
                                      args_init1,
                                      stations,
                                      temporal_aggs,
                                      training_mode,
                                      metric_list,
                                      min_flow = 20,
                                      station = None,
                                      clustered_stations = None,
                                      folder_path = None, 
                                      save_name = None,
                                      bool_plot = True, 
                                      dates = None,
                                      comparison_on_rainy_events = False
                                      ):
    """
    args : 
    ------
        If dates is not None: Only if full_predict1, full_predict2, Y_true and X does not correspond to the full set of the training mode.
        Example of use:  When full_predict is a subset of the full prediction on test-set, associated to rainy dates.
    """
    print('Model1 correspond to : ',trial_id1)
    print('Model2 correspond to : ',trial_id2)

    step_ahead_max = args_init1.step_ahead
    dic_error_agg_h = {}

    # -- Comparison plotting for each horizon : 
    for step_ahead in range(args_init1.horizon_step,step_ahead_max+1,args_init1.horizon_step): # range(1,step_ahead_max+1):   
        h_idx = step_ahead// args_init1.horizon_step
        if save_name is not None:
            save_name_i = save_name.split('_bis')[0]
            if comparison_on_rainy_events:
                save_name_i += "_rainy"
        else:
            save_name_i = None
            
        dic_error_agg_h,real = comparison_plotting(dic_error_agg_h,    
                                                   full_predict1,
                                                   full_predict2,ds1,Y_true,X,temporal_aggs,step_ahead,
                                                   h_idx,stations,training_mode,metric_list,clustered_stations,
                                                   folder_path = folder_path,
                                                   save_name = save_name_i,
                                                   bool_plot = bool_plot,
                                                   dates = dates,
                                                   min_flow = min_flow,)

    # --

    # -- Plot Temporal profil if needed : 
    if station is not None:
        station_i = list(ds1.spatial_unit).index(station)
        p = plot_profile_comparison_between_2_prediction(args_init1,full_predict1,full_predict2,real,ds1,station_i,station, width=900, height=400, bool_plot = bool_plot)
    # --


    # -- Display some informations: 
    display_information_related_to_comparison(dic_error_agg_h,args_init1,metric_list,ds1.spatial_unit,step_ahead_max)
    # --


def load_trainer_ds_from_2_trials(trial_id1,trial_id2,modification,model_args,path_model_args,path_model_args_bis=None,ds1_init=None,
                                ds2_init = None,args_init1 = None, args_init2 = None,model_args_bis=None,
                                trial_id1_in_bis = False, trial_id2_in_bis = False):
    """
    Load trainer and dataset from two trials.
    Will be used to compare the two models.
    """
    print('Trial ID 1: ',trial_id1)
    print('Trial ID 2: ',trial_id2)
    # Sometimes trial_id is given with a unnecessary prefix '_':
    if trial_id1_in_bis :
        print('trial_id1 supposed to be in bis')
        if (trial_id1[1:] in model_args_bis['model'].keys()):
            trial_id1 = trial_id1[1:]
        args = model_args_bis['model'][trial_id1]['args']
        path = path_model_args_bis
    else: 
        if trial_id1[1:] in model_args['model'].keys():
            trial_id1 = trial_id1[1:]
        if not trial_id1 in model_args['model'].keys() and (model_args_bis is not None):
            if (trial_id1[1:] in model_args_bis['model'].keys()):
                trial_id1 = trial_id1[1:]
        if trial_id1 in model_args['model'].keys():
            args = model_args['model'][trial_id1]['args']
            path  = path_model_args
        elif (model_args_bis is not None) and (trial_id1 in model_args_bis['model'].keys()):
            args = model_args_bis['model'][trial_id1]['args']
            path = path_model_args_bis
        else:
            print(list(model_args['model'].keys()))
            raise ValueError(f"Trial ID 1 {trial_id1} not found in model_args or model_args_bis.")
    model_save_path = f"{path}/{trial_id1}.pkl"
    print('model_save_path for trial id1: ',model_save_path)
    trainer1, ds1, args_init1 = load_trainer_ds_from_saved_trial(args,model_save_path,modification = modification,ds_init=ds1_init,args_init = args_init1)
    

    # Trial id dans model_args_bis : 
    if trial_id2_in_bis :
        if (trial_id2[1:] in model_args_bis['model'].keys()):
            trial_id2 = trial_id2[1:]
        args = model_args_bis['model'][trial_id2]['args']
        path = path_model_args_bis
    # Sinon : 
    else: 
        if (trial_id2[1:] in model_args['model'].keys()):
            trial_id2 = trial_id2[1:]
        if trial_id2 in model_args['model'].keys():
            args = model_args['model'][trial_id2]['args']
            path  = path_model_args

        # Si on trouve quand même pas dans model_args normal : 
        elif (model_args_bis is not None):
                if (trial_id2[1:] in model_args_bis['model'].keys()):
                    trial_id2 = trial_id2[1:]
                if trial_id2 in model_args_bis['model'].keys():
                    args = model_args_bis['model'][trial_id2]['args']
                    path = path_model_args_bis
                else:
                    raise ValueError(f"Trial ID 2 {trial_id2} not found in model_args_bis.")
        else:
            raise ValueError(f"Trial ID 2 {trial_id2} not found in model_args")
    
    model_save_path = f"{path}/{trial_id2}.pkl"
    print('model_save_path for trial id2: ',model_save_path)
    try: 
        trainer2, ds2, args_init2 = load_trainer_ds_from_saved_trial(args,model_save_path,modification = modification,ds_init = ds2_init,args_init=args_init2)
    except:
        trainer2, ds2, args_init2 = None, None, None
    return trainer1,trainer2,ds1,ds2,args_init1,args_init2 


def load_trainer_ds_from_1_args(args_init,modification = {},save_folder = None,trial_id = None,fold_to_evaluate=None):
    if fold_to_evaluate is None:
        fold_to_evaluate=[args_init.K_fold-1]
    trainer,ds,model,args = load_init_model_trainer_ds(fold_to_evaluate,save_folder,args_init,modification,trial_id)  
    return trainer,ds 




class ComparisonPlotter:
    """
    Class to plot the gain between two models with different temporal aggregations.
    Updated to force heatmaps to fill space and use a thin, dedicated colorbar axis.
    """
    def __init__(self,
    clustered_stations=None,
    width_ratios= None, 
    fig_size_x = None,
    fig_size_y = None,
    cbar_magic_args = False,
    size_colorbar = "4%",
    ):
        self.clustered_stations = clustered_stations 
        self.width_ratios = width_ratios
        self.size_colorbar = size_colorbar
        self.fig_size_x = fig_size_x
        self.fig_size_y = fig_size_y
        self.cbar_magic_args = cbar_magic_args

    def _aggregated_plot(self, dic_gain21, ax,yaxis):
        ax = plot_coverage_matshow(
            pd.DataFrame(pd.DataFrame(dic_gain21).mean(axis=1)),
            cmap='RdYlBu', save=None, 
            bool_reversed=True, v_min=-10, v_max=10, display_values=True, 
            cbar_magic_args = self.cbar_magic_args,
            ax = ax,
            size_colorbar = self.size_colorbar,
            xaxis = "All units",
            yaxis = yaxis,  
        )
        ax.set_title(f"Aggregated\nthrough\nstations")
        return ax

    def _plot_per_station(self, dic_gain21, ax, metric):
        if self.clustered_stations is None: 
            df_matshow = pd.DataFrame(dic_gain21)
        else:
            df_matshow = pd.DataFrame(dic_gain21)[list(itertools.chain.from_iterable([v for k,v in sorted(self.clustered_stations.items())]))]
            
        ax = plot_coverage_matshow(
            df_matshow,
            cmap='RdYlBu', save=None, 
            cbar_label=f'{metric.upper()} Gain (%)',
            bool_reversed=True, v_min=-20, v_max=20, display_values=False, 
            cbar_magic_args = self.cbar_magic_args,
            ax = ax,
            size_colorbar = self.size_colorbar,
            xaxis = "Spatial Units",
            yaxis = None
        )
        ax.set_title(f"No\nSpatial\nAggregation") 
        return ax

    def _plot_per_cluster(self, dic_gain21, ax):
        if self.clustered_stations is not None: 
            df_gain21 = pd.DataFrame(dic_gain21)
            df_gain_aggregated_per_cluster = pd.DataFrame(index=df_gain21.index)
            for cluster_id, station_list in sorted(self.clustered_stations.items()):
                df_gain_aggregated_per_cluster[cluster_id] = df_gain21[station_list].mean(axis=1)

            ax = plot_coverage_matshow(
                df_gain_aggregated_per_cluster,
                cmap='RdYlBu', save=None, 
                # cbar_label=f'{metric.upper()} Gain (%)',
                bool_reversed=True, v_min=-10, v_max=10, display_values=True, 
                cbar_magic_args = self.cbar_magic_args,
                ax = ax,
                size_colorbar = self.size_colorbar,
                xaxis = "Clusters",
                yaxis = None
            )
            ax.set_title(f"Aggregated\nthrough\ncluster")
            return ax,df_gain_aggregated_per_cluster
        else:
            return None
        


    def plot_gain_between_models_with_temporal_agg(self,
                                                   ds,
                                                   dic_error, training_mode,stations, dates,temporal_aggs,
                                                   metric_list,
                                                   bool_plot=True,
                                                   save_name = None,
                                                   folder_path = None,
                                                   ):
        self.bool_plot = bool_plot
        self.dic_gain_agg = {metric: {} for metric in metric_list}
        self.dic_error_agg = {metric: {} for metric in metric_list}
        self.dic_gain_agg_per_cluster = {metric: {} for metric in metric_list}



        # ---- Plot Error metric but no gain: 
        self._plot_error_per_day(ds,dic_error,metric_list,training_mode,temporal_agg='date',stations=stations,dates=dates,folder_path=folder_path,save_name=save_name)

        # ---- Save .csv metrics on specific temporal aggs:  

        # self._init_fig_axes_sizes(temporal_aggs, len(stations))
        for metric in metric_list:
            for i, temporal_agg in enumerate(temporal_aggs):
                if temporal_agg == 'working_day_hour':
                    str_temporal_agg = 'Hour on Business Day'
                else: 
                    str_temporal_agg  = temporal_agg
                
                if temporal_agg == 'working_day_hour':
                    yaxis = 'Hour'
                else:
                    yaxis = str_temporal_agg


                title = f"Average {metric.upper()} Gain(%) per {str_temporal_agg}" #  at (H{step_ahead})"

                if metric == 'mase':
                    dic_gain21,error_pred1_agg,error_pred2_agg = get_df_mase_and_gains(ds,dic_error,training_mode,temporal_agg,stations,dates=dates)
                else:
                    dic_gain21,error_pred1_agg,error_pred2_agg = get_df_gains(ds,dic_error,metric,training_mode,temporal_agg,stations,dates=dates)

                self.dic_gain_agg[metric][temporal_agg] = dic_gain21
                self.dic_error_agg[metric][temporal_agg] = {'error_pred1_agg': error_pred1_agg, 'error_pred2_agg': error_pred2_agg}

                cols = 3 if self.clustered_stations is not None else 2
                ratios = self.width_ratios if cols == 3 else self.width_ratios[:2]
                _, axes = plt.subplots(1, cols, 
                            figsize=(self.fig_size_x, self.fig_size_y),
                            gridspec_kw={'width_ratios': ratios}
                            )

                ax1 = self._aggregated_plot(dic_gain21, axes[0],yaxis )
                ax2 = self._plot_per_station(dic_gain21, axes[1],metric)
                if cols == 3:
                    ax3,df_gain_aggregated_per_cluster = self._plot_per_cluster(dic_gain21, axes[2])
                    self.dic_gain_agg_per_cluster[metric][temporal_agg] = df_gain_aggregated_per_cluster
                else:
                    ax3 = None  


                layouts = [ax1,ax2] if ax3 is None else [ax1,ax2,ax3]

                if save_name is not None:
                    if 'rainy' in save_name:
                        desag_name = 'desag_rainy'
                    else:
                        desag_name = 'desag'
                    save_path = f"{folder_path}/{desag_name}/{metric}/{save_name}_gain"
                    if not os.path.exists(f"{folder_path}/{desag_name}/"):
                        os.mkdir(f"{folder_path}/{desag_name}/")
                    if not os.path.exists(f"{folder_path}/{desag_name}/{metric}/"):
                        os.mkdir(f"{folder_path}/{desag_name}/{metric}/")
                else:
                    save_path = None
                self.deal_with_subplots(layouts,title,save_path,save_name)

    def deal_with_subplots(self,layouts,title,save_path,save_name):
        """ Align subplots horizontally"""
        if len(layouts) > 0 and layouts[0] is not None:
            fig = layouts[0].figure
            fig.suptitle(title, fontsize=14, y=1.02)
            plt.tight_layout()
            plt.show()
            if save_path is not None:
                try: 
                    print(f"Figure saved in {save_path}.pdf")
                    fig.savefig(f"{save_path}.pdf", bbox_inches='tight')
                except:
                    print(f"Figure saved in {save_name}.pdf")
                    fig.savefig(f"{save_name}.pdf", bbox_inches='tight')


    def _plot_error_per_day(self,ds,dic_error,metric_list,training_mode,temporal_agg,stations,dates=None,folder_path = None,save_name = None):

        for metric in metric_list:
            dic_gain21,error_pred1_agg,error_pred2_agg = get_df_gains(
                                ds,
                                dic_error,
                                metric,
                                training_mode,
                                temporal_agg ='date',
                                stations=stations,
                                dates = dates)
            df_mae = pd.DataFrame(pd.DataFrame(error_pred2_agg).mean(axis=1), columns = [metric])
            df_mae.index = pd.to_datetime(df_mae.index)

            df_mae['weekday_name'] = df_mae.index.day_name()
            iso_calendar = df_mae.index.isocalendar()
            df_mae['week_of_year_year'] = iso_calendar.year.astype(str) + ' - ' + iso_calendar.week.astype(str)
            df_mae_pivot = df_mae.pivot(index='week_of_year_year', columns='weekday_name', values='mae')
            df_mae_pivot =  df_mae_pivot[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]

            def convert_week_to_date(str_week):
                # Correction : inversion de l'ordre et conversion en int
                year_num, week_num  = str_week.split(' - ')
                date_obj = datetime.date.fromisocalendar(int(year_num), int(week_num), 1)
                return date_obj.strftime("%d %B %Y")

            df_mae_pivot =  df_mae_pivot[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
            df_mae_pivot = df_mae_pivot.sort_index(ascending=True)
            df_mae_pivot.index = df_mae_pivot.index.map(convert_week_to_date)

            fig, ax = plt.subplots(1, 1, 
                        figsize=(6, 10),
                        )
            
            ax = plot_coverage_matshow(
                            df_mae_pivot,
                            cmap='cool', save=None, 
                            bool_reversed=False,
                            # v_min=0, 
                            # v_max=10, 
                            display_values=False, 
                            cbar_magic_args = self.cbar_magic_args,
                            size_colorbar = self.size_colorbar,
                            xaxis = "Weekday",
                            yaxis = "Day - Month", 
                            ax = ax
                        )
            ax.set_title(f"{metric.upper()} on each day")
            plt.show()
            if save_name is not None:
                if 'rainy' in save_name:
                    desag_name = f'error_per_day_rainy'
                else:
                    desag_name = f'error_per_day'
                save_path = f"{folder_path}/{desag_name}/{save_name}"
                if not os.path.exists(f"{folder_path}/{desag_name}/"):
                    os.mkdir(f"{folder_path}/{desag_name}/")

                save_path_i = f"{save_path}_{metric}"
                print(f"Figure saved in {save_path_i}.pdf")
                fig.savefig(f"{save_path_i}.pdf", bbox_inches='tight')
        # ------






def get_rainy_indices(args,ds,training_mode = 'test'):
  # Load Weather DF : 
  df_weather = load_preprocessed_weather_df(args= args,coverage_period=pd.date_range(start=START_weather, end=END_weather, freq=args.freq)[:-1],folder_path=FOLDER_PATH)

  # Test time slots: 
  time_slots = getattr(ds.tensor_limits_keeper,f"df_verif_{training_mode}").iloc[:,-1]
  if (training_mode=='train') and hasattr(args,'expanding_train') and (args.expanding_train is not None) and (args.expanding_train != 1):
    n = len(time_slots)
    split = int(n * args.expanding_train)
    time_slots = time_slots.iloc[-split:]
  total_indices = len(time_slots)

  time_slots.name = 'timestamp'
  df_time_slots = pd.DataFrame(time_slots) 
  df_time_slots = df_time_slots.reset_index(drop=True)
  df_time_slots['indices'] = df_time_slots.index
  df_time_slots = df_time_slots.set_index('timestamp')

  # Rainy Mask : 
  rainy_mask = (df_weather > 0)
  rainy_mask = rainy_mask.loc[time_slots,:].any(axis=1)

  # Extract Rainy Time Slots and rainy indices : 
  rainy_indices = df_time_slots[rainy_mask].indices
  rainy_indices = torch.Tensor(rainy_indices.values).long()

  if training_mode == 'train':
    print(f"Number of rainy time-slots in the train set:")
    L_rainfall = [0,0.05,0.5,1,np.inf]
    for pos in range(len(L_rainfall)-1):
        mask_i = (df_weather > L_rainfall[pos]) & (df_weather <= L_rainfall[pos+1])
        mask_i = mask_i.loc[time_slots,:].any(axis=1)
        indices_i = df_time_slots[mask_i].indices
        indices_i = torch.Tensor(indices_i.values).long()
        print(f" Between {L_rainfall[pos]} and {L_rainfall[pos+1]} mm: {len(indices_i)}, i.e {len(indices_i)/total_indices*100:.2f} % of the train set and {len(indices_i)/len(rainy_indices)*100:.2f} % of the rainy time-slots")


  return rainy_mask,rainy_indices,df_weather,total_indices


def get_cluster(df,
                temporal_agg='business_day', 
                normalisation_type ='minmax',
                index='Station',
                city='Lyon',
                n_clusters=5, 
                linkage_method='complete', 
                metric='precomputed',
                min_samples=2,
                heatmap= True, 
                daily_profile=True, 
                dendrogram=True,
                bool_plot = False,
                folder_path= None,
                save_name = None,
                ):
    # Get Clustering of stations from these inputs: 
    clusterer = TimeSeriesClusterer(df)
    clusterer.preprocess(temporal_agg=temporal_agg, normalisation_type =normalisation_type,index= index,city=city) # 'morning','evening','morning_peak','evening_peak','off_peak','non_business_day','business_day'
    clusterer.run_agglomerative(n_clusters=n_clusters, linkage_method=linkage_method, metric=metric,min_samples=min_samples)
    # clusterer.run_agglomerative(n_clusters=None, linkage_method='complete', metric='precomputed',min_samples=4,distance_threshold = 0.35)
    clusterer.plot_clusters(heatmap= heatmap, 
                            daily_profile=daily_profile,
                            dendrogram=dendrogram,
                            bool_plot = bool_plot,
                            folder_path= folder_path,
                            save_name = save_name,
                            )
    return clusterer

def get_model_args(save_folder_name = 'optim/subway_in_STGCN', save_folder_name_bis = 'optim/config/subway_in_STGCN' ):
    subfolder = f'K_fold_validation/training_wo_HP_tuning/{save_folder_name}'
    path_model_args = f"{SAVE_DIRECTORY}/{subfolder}/best_models"
    model_args = pickle.load(open(f"{path_model_args}/model_args.pkl", 'rb'))
    if save_folder_name_bis is not None: 
        subfolder_bis = f'K_fold_validation/training_wo_HP_tuning/{save_folder_name_bis}'
        path_model_args_bis = f"{SAVE_DIRECTORY}/{subfolder_bis}/best_models"
        model_args_bis = pickle.load(open(f"{path_model_args_bis}/model_args.pkl", 'rb'))
    else:
        subfolder_bis,path_model_args_bis,model_args_bis = None, None, None
    return model_args,model_args_bis,path_model_args,path_model_args_bis

def get_full_predict_on_test_set(trial_id1,
                                 trial_id2,
                                 training_mode,
                                 range_k, 
                                 model_args,
                                 path_model_args,
                                 model_args_bis = None,
                                 path_model_args_bis = None,
                                 trial_id1_in_bis=False,
                                 trial_id2_in_bis=False,
                                 modification = {
                                        'shuffle':False,
                                        'data_augmentation':False,
                                        'torch_compile': False,
                                        }
                                ):
    """
    Get the full predictions on the test set for two configurations (trial_id1 and trial_id2) on each of the range_k runs.

    args:
    -----
    trial_id1: str, trial ID of the first model configuration.
    trial_id2: str, trial ID of the second model configuration.
    ds1: DataSet object,  associated to the first model configuration.
    ds2: DataSet object,  associated to the second model configuration.
    training_mode: str, mode for training, e.g., 'train', 'val', 'test'.
    range_k: iterable, range of runs to consider for loading the trainers and datasets (e.g., range(1,6) for runs 1 to 5).
    modification: if needed. by default, no shuffle, no data augmentation, and no torch compile


    return :
    ------
    full_predict1: Tensor of shape (n_samples, n_stations, n_horizons, n_runs) containing the predictions of model 1 for each run in range_k.
    full_predict2: Tensor of shape (n_samples, n_stations, n_horizons, n_runs) containing the predictions of model 2 for each run in range_k.
    """
    ds1,ds2,args_init1,args_init2 = None, None, None, None

    # -------- Load trainers, datasets, predictions and real values for each k in range_k:
    for k in range_k:
        trial_id1_updated = f"_{trial_id1}{k}_f5"
        trial_id2_updated = f"_{trial_id2}{k}_f5"
        trainer1,trainer2,ds1,ds2,args_init1,args_init2 = load_trainer_ds_from_2_trials(
            trial_id1_updated,
            trial_id2_updated,
            modification = modification,
            model_args=model_args,
            path_model_args=path_model_args,
            path_model_args_bis = path_model_args_bis,
            ds1_init=ds1,ds2_init=ds2,
            args_init1=args_init1,args_init2=args_init2,
            model_args_bis = model_args_bis,
            trial_id1_in_bis = trial_id1_in_bis, 
            trial_id2_in_bis = trial_id2_in_bis
            )
        globals()[f"trainer1_bis{k}"] = trainer1
        globals()[f"trainer2_bis{k}"] = trainer2
        globals()[f"ds1_bis{k}"] = ds1
        globals()[f"ds2_bis{k}"] = ds2

        if trainer2 is not None:                                             
            full_predict1,full_predict2,Y_true,X = get_predict_real_and_inputs(trainer1,trainer2,ds1,ds2,training_mode=training_mode)

            globals()[f"full_predict1_bis{k}"] = full_predict1
            globals()[f"full_predict2_bis{k}"] = full_predict2
        else:
            break

    full_predict1 = torch.stack([globals()[f"full_predict1_bis{k}"] for k in range_k],-1)
    full_predict2 = torch.stack([globals()[f"full_predict2_bis{k}"] for k in range_k],-1)

    return [globals()[f"trainer1_bis{k}"] for k in range_k],[globals()[f"trainer2_bis{k}"] for k in range_k],full_predict1,full_predict2,ds1,ds2,args_init1,args_init2,Y_true,X



def get_desagregated_comparison_plot(trial_id1,trial_id2,
                                     model_args,
                                     model_args_bis,
                                     path_model_args,
                                     path_model_args_bis,
                                     range_k = range(1,6),
                                    trial_id1_in_bis=False,
                                    trial_id2_in_bis=False,
                                    colmumn_name = 'Station',
                                    comparison_on_rainy_events = False,
                                    station_clustering = False,
                                    folder_path = None,
                                    save_name = None,
                                    heatmap = False,
                                    daily_profile = False,
                                    dendrogram = False,
                                    bool_plot = False,
                                    clusters = None,
                                    list_top_k_percent = [None],
                                    ):


    training_mode = 'test'

    L_trainer1,L_trainer2,full_predict1,full_predict2,ds1,ds2,args_init1,args_init2,Y_true,X = get_full_predict_on_test_set(
                                                                                                trial_id1=trial_id1,
                                                                                                trial_id2=trial_id2,
                                                                                                training_mode=training_mode,
                                                                                                range_k=range_k, 
                                                                                                model_args=model_args,
                                                                                                path_model_args=path_model_args,
                                                                                                model_args_bis = model_args_bis,
                                                                                                path_model_args_bis = path_model_args_bis,
                                                                                                trial_id1_in_bis= trial_id1_in_bis,
                                                                                                trial_id2_in_bis = trial_id2_in_bis,
                                                                                                modification = {
                                                                                                    'shuffle':False,
                                                                                                    'data_augmentation':False,
                                                                                                    'torch_compile': False,
                                                                                                    }
                                                                                                )
    trainer2 = L_trainer2[-1]
    if trainer2 is None:
        return None,None,None,None,None,None,None,None,None,None,None,None,None


    dic_bd_metrics = {trial_id1: {int(topk_percent*100) if topk_percent is not None else 100 : {} for topk_percent in list_top_k_percent},
                      trial_id2: {int(topk_percent*100) if topk_percent is not None else 100 : {} for topk_percent in list_top_k_percent},
                                }
    
    # --- For Each Top K Percent: 
    for topk_percent in list_top_k_percent :
        # 1. --- For each top-k percent, filter train_input to keep only the top-k percent of stations with the highest flow:
        train_input = ds2.train_input.clone()
        train_time_slots = ds2.tensor_limits_keeper.df_verif_train.stack().unique()
        train_input = pd.DataFrame(train_input.numpy(),index = train_time_slots,columns = ds2.spatial_unit)
        train_input = train_input.reindex(pd.date_range(start=train_input.index.min(),end=train_input.index.max(),freq=args_init2.freq))
        train_input.columns.name = colmumn_name
        filtered_train_input,top_ids = get_topk(train_input,topk_percent)

        # ---- Set save path: ---
        if topk_percent is not None:
            folder_path_i = f"{folder_path}/top{int(topk_percent*100)}percent" if folder_path is not None else None
        else:
            folder_path_i = folder_path
            
        if folder_path_i is not None and not os.path.exists(folder_path_i):
            os.mkdir(folder_path_i)
        # ----
        
        # 2. --- If clustering, Get Cluster : 
        if station_clustering :
            # Get Clustering of stations from these inputs:
            clusterer = get_cluster(filtered_train_input,
                                    temporal_agg='business_day',
                                    normalisation_type ='minmax',
                                    index= colmumn_name,
                                    city=ds2.city,
                                    n_clusters=5, 
                                    linkage_method='complete', 
                                    metric='precomputed',
                                    min_samples=2,
                                    heatmap= heatmap, 
                                    daily_profile=daily_profile, 
                                    dendrogram=dendrogram,
                                    bool_plot = bool_plot,
                                    folder_path= folder_path_i,
                                    save_name = save_name,
                                    )
        elif clusters is not None:
            clusterer = lambda : None
            clusterer.clusters = clusters
            train_input = None
        else:
            clusterer = lambda : None
            clusterer.clusters = None
            train_input = None
        # ------------------

        list_rainy = [True, False] if comparison_on_rainy_events else [False]

        for bool_rainy in list_rainy:
            # Temporal agg : rainy or not rainy (by default business day on desag)

            if bool_rainy: 
                print("\nComparison on between models across all time-slots followed by comparison on Rainy Events Only")
                _,train_rainy_indices,_,_ = get_rainy_indices(args = args_init2,ds = ds2,training_mode = 'train')
                print(f"Number of rainy time-slots in the train set: {len(train_rainy_indices)}, i.e {len(train_rainy_indices)/len(ds2.tensor_limits_keeper.df_verif_train)*100:.2f} % of the train set")
                # ---- Plot Accuracy comparison on rainy moments only ----
                mask,temporal_agg_indices,df_weather,_ = get_rainy_indices(args = args_init2,ds = ds2,training_mode = 'test')
                print(f"Number of rainy time-slots in the test set: {len(temporal_agg_indices)}, i.e {len(temporal_agg_indices)/len(ds2.tensor_limits_keeper.df_verif_test)*100:.2f} % of the test set\n")
                dates_i = mask[mask].index


            else:
                temporal_agg_indices = None
                dates_i = getattr(ds2.tensor_limits_keeper,f"df_verif_{training_mode}").iloc[:,-1]


            specific_nodes = torch.Tensor([list(ds1.spatial_unit).index(station) for station in top_ids]).long() if top_ids is not None else None

            full_predict1_i,full_predict2_i,Y_true_i,_,X_i = filter_predictions_and_real_values(full_predict1,full_predict2,Y_true,
                                                                                        specific_indices = temporal_agg_indices,
                                                                                        specific_nodes = specific_nodes,
                                                                                        X = X,
                                                                                        )

            metric_list   = ['mae'] # ['mae','mase','rmse']
            temporal_aggs = ['working_day_hour']

            plot_analysis_comparison_2_config(trial_id1,trial_id2,
                                            full_predict1_i,
                                            full_predict2_i,
                                            Y_true_i,
                                            X_i,
                                            ds1,
                                            args_init1,
                                            top_ids,  # (old) stations 
                                            temporal_aggs,
                                            training_mode,
                                            metric_list,
                                            min_flow = 20,
                                            station = None,
                                            clustered_stations = clusterer.clusters,
                                            dates = dates_i,
                                            folder_path = folder_path_i,
                                            save_name = save_name,
                                            comparison_on_rainy_events = bool_rainy
                                            )

            for local_temporal_agg in [None, 'business_day']:
                if local_temporal_agg is not None: 
                    s_dates = ds2.tensor_limits_keeper.df_verif_test.iloc[:,-1].reset_index(drop=True)
                    dates_ii = pd.Series(dates_i).reset_index(drop=True)
                    local_temporal_agg_indices = get_temporal_mask(s_dates = dates_ii, # s_dates,
                                    start = datetime.time(0, 00),
                                    end=datetime.time(23, 59),
                                    temporal_agg = local_temporal_agg,
                                    city = ds2.city,
                    )
                    local_temporal_agg_indices = torch.Tensor(local_temporal_agg_indices.values).long()
                else:
                    local_temporal_agg_indices = temporal_agg_indices


                RMSE1,MAE1,MASE1,MAPE1,RMSE2,MAE2,MASE2,MAPE2 = get_str_metrics(
                        args_init2,
                        X,
                        Y_true,
                        full_predict1,
                        full_predict2,
                        specific_indices=local_temporal_agg_indices,
                        specific_nodes = specific_nodes,
                    )
                
                # ----- Fill dic_bd_metrics :
                key_topk = 100 if topk_percent is None else int(topk_percent*100)
                key_temporal_agg = local_temporal_agg if local_temporal_agg is not None else 'all'
                if bool_rainy:
                    if not 'rainy' in dic_bd_metrics[trial_id1][key_topk].keys():
                        dic_bd_metrics[trial_id1][key_topk]['rainy'] = {}
                    dic_bd_metrics[trial_id1][key_topk]['rainy'][key_temporal_agg] = {'RMSE': RMSE1, 'MAE': MAE1, 'MASE': MASE1, 'MAPE': MAPE1}

                    if not 'rainy' in dic_bd_metrics[trial_id2][key_topk].keys():
                        dic_bd_metrics[trial_id2][key_topk]['rainy'] = {}
                    dic_bd_metrics[trial_id2][key_topk]['rainy'][key_temporal_agg] = {'RMSE': RMSE2, 'MAE': MAE2, 'MASE': MASE2, 'MAPE': MAPE2}
                else:
                    if not 'all' in dic_bd_metrics[trial_id1][key_topk].keys():
                        dic_bd_metrics[trial_id1][key_topk]['all'] = {}
                    dic_bd_metrics[trial_id1][key_topk]['all'][key_temporal_agg] = {'RMSE': RMSE1, 'MAE': MAE1, 'MASE': MASE1, 'MAPE': MAPE1}
                    
                    if not 'all' in dic_bd_metrics[trial_id2][key_topk].keys():
                        dic_bd_metrics[trial_id2][key_topk]['all'] = {}
                    dic_bd_metrics[trial_id2][key_topk]['all'][key_temporal_agg] = {'RMSE': RMSE2, 'MAE': MAE2, 'MASE': MASE2, 'MAPE': MAPE2}
                # -----

    

    return clusterer,full_predict1,full_predict2,train_input,X,Y_true,L_trainer1,L_trainer2, ds1,ds2,args_init1,args_init2,dic_bd_metrics


if __name__ == '__main__':

    from constants.paths import SAVE_DIRECTORY
    ## -----------------FULL DATA 1 AN---------------------------------------------------------------------------------------------------------
    ## Prediction on 4 consecutives horizons 

    # ## Prediction on HORIZON 1
    # trial_id1 = 'subway_out_calendar_embedding_h1_bis'
    # trial_id2 = 'subway_out_subway_in_bike_in_calendar_embedding_h1_bis'
    # trial_id2 = 'subway_out_subway_in_calendar_embedding_h1_bis'
    # trial_id2 = 'subway_out_bike_in_calendar_embedding_h1_bis'

    # ## Prediction on HORIZON 2
    # trial_id1 = 'subway_out_calendar_embedding_h2_bis'
    # trial_id2 = 'subway_out_subway_in_bike_in_calendar_embedding_h2_bis'
    # trial_id2 = 'subway_out_subway_in_calendar_embedding_h2_bis'
    # trial_id2 = 'subway_out_bike_in_calendar_embedding_h2_bis'

    # ## Prediction on HORIZON 3
    # trial_id1 = 'subway_out_calendar_embedding_h3_bis'
    # trial_id2 = 'subway_out_subway_in_bike_in_calendar_embedding_h3_bis'
    # trial_id2 = 'subway_out_subway_in_calendar_embedding_h3_bis'
    # trial_id2 = 'subway_out_bike_in_calendar_embedding_h3_bis'

    # ## Prediction on HORIZON 4
    # trial_id1 = 'subway_out_calendar_embedding_h4_bis'
    # trial_id2 = 'subway_out_subway_in_bike_in_calendar_embedding_h4_bis'
    # trial_id2 = 'subway_out_subway_in_calendar_embedding_h4_bis'
    # trial_id2 = 'subway_out_bike_in_calendar_embedding_h4_bis'
    

    # --- Init
    model_name = 'STAEformer' # 'STGCN', 'STAEformer'
    if model_name == 'STGCN':
        calendar_str = 'calendar_embedding'
    if model_name == 'STAEformer':
        calendar_str = 'calendar'
    SAVE_DIRECTORY = f"{SAVE_DIRECTORY}"
    subfolder = f'K_fold_validation/training_wo_HP_tuning/optim/subway_in_{model_name}'
    path_model_args = f"{SAVE_DIRECTORY}/{subfolder}/best_models"
    model_args = pickle.load(open(f"{path_model_args}/model_args.pkl", 'rb'))



    target_data = 'subway_in'# 'subway_in' #'subway_out'
    subway_data_2 = 'subway_out' # 'subway_out' #'subway_in'


    trial_id1_i = f"{target_data}"
    trial_id1_ii = f"{target_data}_{subway_data_2}"

    dic_contextual_datasets = { 
                                trial_id1_i: [f'{subway_data_2}',
                                              f'bike_out',f'bike_out_{subway_data_2}'
                                              ],

                                # trial_id1_i: [f'{subway_data_2}_bike_in_bike_out',
                                #               f'{subway_data_2}_bike_in',f'{subway_data_2}_bike_out', f'bike_in_bike_out',
                                #               f'{subway_data_2}',f'bike_in',f'bike_out'],

                                # trial_id1_ii: [f'{subway_data_2}_bike_in_bike_out',
                                #                f'{subway_data_2}_bike_in',f'{subway_data_2}_bike_out', f'bike_in_bike_out',
                                #                f'bike_in',f'bike_out']
                                trial_id1_ii : [f'bike_in_bike_out']
                            }
    # ---

    
    for trial_id1_init in [trial_id1_i]: # [trial_id1_i,trial_id1_ii]: # [trial_id1_i]:
        contextual_datasets = dic_contextual_datasets[trial_id1_init]

        for contextual_dataset in contextual_datasets:
            for horizon in [1,4]: #[1,2,3,4]: 
              

                trial_id1 = f'{trial_id1_init}_{calendar_str}_h{horizon}_bis'
                trial_id2 = f'{target_data}_{contextual_dataset}_{calendar_str}_h{horizon}_bis'



                INIT_SAVE_PATH = f"{SAVE_DIRECTORY}/plot/comparison_between_models/{model_name}_prediction_{target_data}"

                # save_name = f"{trial_id2}"
                save_name = f"ref_{trial_id1_init}_h{horizon}_to_{trial_id2[:-4]}"

                print('\n--------------------\nINIT_SAVE_PATH: ',INIT_SAVE_PATH)
                print('save_name: ',save_name)



                modification = {'shuffle':False,
                            'data_augmentation':False,
                            'torch_compile': False,
                            'device': torch.device('cuda:0')
                            }
                training_mode = 'test'
                temporal_aggs =   ['working_day_hour'] # ['daily_period','working_day_hour','weekday_hour_minute'] # ['hour','date','weekday'] 'hour', 'date', 'weekday', 'weekday_hour', 'weekday_hour_minute', 'daily_period', 'working_day_hour'
                metric_list = ['mae','rmse'] # ['mae','mase','rmse']

                #  ----  Load saved models and predictions  ---- 
                ds1,args_init1 = None,None
                ds2,args_init2 = None,None
                range_k = range(1,6)   # 5 trials per config
                for k in range_k: # range(1,6):
                    trial_id1_updated = f"_{trial_id1}{k}_f5"
                    trial_id2_updated = f"_{trial_id2}{k}_f5"
                    trainer1,trainer2,ds1,ds2,args_init1,args_init2 = load_trainer_ds_from_2_trials(trial_id1_updated,trial_id2_updated,modification = modification,model_args=model_args,path_model_args=path_model_args,
                                                                                                    ds1_init=ds1,ds2_init=ds2,
                                                                                                    args_init1=args_init1,args_init2=args_init2)
                    full_predict1,full_predict2,Y_true,X = get_predict_real_and_inputs(trainer1,trainer2,ds1,ds2,training_mode=training_mode)
                    globals()[f"trainer1_bis{k}"] = trainer1
                    globals()[f"trainer2_bis{k}"] = trainer2
                    globals()[f"ds1_bis{k}"] = ds1
                    globals()[f"ds2_bis{k}"] = ds2
                    globals()[f"full_predict1_bis{k}"] = full_predict1
                    globals()[f"full_predict2_bis{k}"] = full_predict2
                

                full_predict1 = torch.stack([globals()[f"full_predict1_bis{k}"] for k in range_k]).mean(0)
                full_predict2 = torch.stack([globals()[f"full_predict2_bis{k}"] for k in range_k]).mean(0)
                stations = list(ds1.spatial_unit)  # ['BEL','PER','PAR','AMP','FOC'] #list(ds1.spatial_unit)

                # ---- 

                # --- Get Cluster : 
                # Load Train inputs: 
                if horizon == 1: 
                    train_input = ds2.train_input
                    train_time_slots = ds2.tensor_limits_keeper.df_verif_train.stack().unique()
                    train_df = pd.DataFrame(train_input.numpy(),index = train_time_slots,columns = ds2.spatial_unit)
                    train_df = train_df.reindex(pd.date_range(start=train_df.index.min(),end=train_df.index.max(),freq='15min'))

                    # Get Clustering of stations from these inputs: 
                    clusterer = TimeSeriesClusterer(train_df)
                    clusterer.preprocess(temporal_agg='business_day', normalisation_type ='minmax',index= 'Station',city=ds2.city) # 'morning','evening','morning_peak','evening_peak','off_peak','non_business_day','business_day'
                    clusterer.run_agglomerative(n_clusters=5, linkage_method='complete', metric='precomputed',min_samples=2)
                    # clusterer.run_agglomerative(n_clusters=None, linkage_method='complete', metric='precomputed',min_samples=2,distance_threshold = 0.1)

                clusterer.plot_clusters(heatmap= True, daily_profile=True, dendrogram=True,folder_path = INIT_SAVE_PATH, save_name = save_name ,bool_plot = False)
                # ---


                # ---- Plot Accuracy Comparison ---- 
                plot_analysis_comparison_2_config(trial_id1,trial_id2,full_predict1,full_predict2,Y_true,X,ds1,args_init1,
                                                    stations,temporal_aggs,training_mode,metric_list,min_flow = 20,station = None,
                                                    clustered_stations = clusterer.clusters,folder_path = INIT_SAVE_PATH, save_name = save_name,bool_plot = False)
                # ----



def filter_predictions_and_real_values(full_predict1,full_predict2,Y_true,
                                       specific_indices=None,
                                       specific_nodes = None,
                                       previous = None,
                                       X = None,
                                       ):
    """
    Select specific time slots (specific_indices) and/or specific nodes (specific_nodes) in differnt tensors: 
    - full_predict1 : (time_slots, nodes, 1, n_bis)
    - full_predict2 : (time_slots, nodes, 1, n_bis)
    - Y_true : (time_slots, nodes, 1)
    - previous : (time_slots, nodes, 1)
    """
    full_predict1_i = full_predict1.clone()
    full_predict2_i = full_predict2.clone()
    Y_true_i = Y_true.clone()
    previous_i = previous.clone() if previous is not None else None
    X_i = X.clone() if X is not None else None

    if specific_indices is not None :
        if type(specific_indices) == list:
            specific_indices = torch.Tensor(specific_indices).long()
        full_predict1_i = torch.index_select(full_predict1_i,0,specific_indices)
        full_predict2_i = torch.index_select(full_predict2_i,0,specific_indices)
        Y_true_i = torch.index_select(Y_true_i,0,specific_indices)
        previous_i = torch.index_select(previous_i,0,specific_indices) if previous_i is not None else None
        X_i = torch.index_select(X_i,0,specific_indices) if X_i is not None else None
    

    if specific_nodes is not None :  
        if type(specific_nodes) == list:
            specific_nodes = torch.Tensor(specific_nodes).long()
        full_predict1_i = torch.index_select(full_predict1_i,1,specific_nodes)
        full_predict2_i = torch.index_select(full_predict2_i,1,specific_nodes)
        Y_true_i = torch.index_select(Y_true_i,1,specific_nodes)
        previous_i = torch.index_select(previous_i,1,specific_nodes) if previous_i is not None else None
        X_i = torch.index_select(X_i,1,specific_nodes) if X_i is not None else None

    return full_predict1_i,full_predict2_i,Y_true_i,previous_i,X_i

def get_str_metrics(args_init2,X,Y_true,full_predict1,full_predict2,
                    specific_indices=None,
                    specific_nodes = None,):
    h_idx = args_init2.step_ahead // args_init2.horizon_step
    assert h_idx == 1, "Error: the function get_str_metrics is currently only implemented for only one step ahead (h_idx = 1), please set step_ahead equal to horizon_step in the model args or implement the get_str_metrics function for h_idx > 1"

    previous = get_previous(X,Y_true,h_idx)
    full_predict1_i,full_predict2_i,Y_true_i,previous_i,_ = filter_predictions_and_real_values(full_predict1,full_predict2,Y_true,specific_indices,specific_nodes,previous=previous)


    RMSE1,MAE1,MASE1,MAPE1 = [],[],[],[]
    RMSE2,MAE2,MASE2,MAPE2 = [],[],[],[]

    for n_bis in range(full_predict1_i.size(-1)):
        full_predict1_i_bis = full_predict1_i[...,n_bis] 
        full_predict2_i_bis = full_predict2_i[...,n_bis] 
        dic_metric1_i = evaluate_metrics(full_predict1_i_bis,Y_true_i,metrics = ['rmse','mse','mae','mase','mape'], previous = previous_i,horizon_step = h_idx)
        dic_metric2_i = evaluate_metrics(full_predict2_i_bis,Y_true_i,metrics = ['rmse','mse','mae','mase','mape'], previous = previous_i,horizon_step = h_idx)
        RMSE1.append(dic_metric1_i['rmse_all'])
        MAE1.append(dic_metric1_i['mae_all'])
        MASE1.append(dic_metric1_i['mase_all'])
        MAPE1.append(dic_metric1_i['mape_all'])
        RMSE2.append(dic_metric2_i['rmse_all'])
        MAE2.append(dic_metric2_i['mae_all'])
        MASE2.append(dic_metric2_i['mase_all'])
        MAPE2.append(dic_metric2_i['mape_all'])

    return RMSE1,MAE1,MASE1,MAPE1,RMSE2,MAE2,MASE2,MAPE2