from bokeh.plotting import figure
from bokeh.models import Legend,BoxAnnotation,DatetimeTickFormatter,RangeTool
import torch 
from datetime import timedelta
import pandas as pd 
from bokeh.palettes import Set3_12 
from bokeh.palettes import Plasma256 
from bokeh.palettes import Turbo256 as palette

from bokeh.plotting import output_notebook,show
import numpy as np 


import torch.nn as nn
import torch
import sys
import os
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from utils.metrics import error_along_ts


def plot_subway_patterns(df,Metro_A_stations,palette,width=1500, height=600,title=f'Trafic Volume by stations'):

    p = figure(x_axis_type="datetime", 
               title=title,
               width=width,height=height)

    legend_it = []


    for k,station in enumerate(Metro_A_stations):
        c = p.line(x=df.index, line_width = 2.5, y=df[station], alpha=0.8,  #legend_label = f'{station}',
                   color=palette[int(k*(256-1)/(len(Metro_A_stations)+1))]
                  )#,muted_color=color, muted_alpha=0.2
        legend_it.append((station, [c]))
    
    p.xaxis.major_label_orientation = 1.2  # Pour faire pivoter les labels des x
    legend = Legend(items=legend_it)
    legend.click_policy="hide"
    p.add_layout(legend, 'right')
    
    return(p)

def drag_selection_box(df,p1,p2=None,p3=None, width=1500, height=150):

    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    width=width,
                    height=height, 
                    y_range=p1.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")

    range_tool = RangeTool(x_range=p1.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select.line(x=df.index, y=df.mean(axis=1))
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    
    # Sync x_range for both plots:
    if p2 is not None:
        p2.x_range = p1.x_range
    if p3 is not None:
        p3.x_range = p1.x_range
    return(select)


def plot_single_point_prediction(df_true,df_prediction,station,title = '',kick_off_time = [], range = None,width=1500,height=400,bool_show=False):
       '''
       args:
       ------
       station:  str or list of str. each elmt of the list represent a spatial unit of the predicted dataset.
       '''

       legend_it = []
       p = figure(x_axis_type="datetime", title= title,
                     width=width,height=height)

       if type(station) != list:
             station = [station]
       
       for k,station_i in enumerate(station):
              c = p.line(x=df_true.index, line_width = 2.5, y=df_true[station_i], alpha=0.8,color = Plasma256[int(k*255/len(station))])
              legend_it.append((f'True_{station_i}', [c]))

              if df_prediction is not None: 
                     if type(df_prediction) == list:
                            for q_i,df_pred in enumerate(df_prediction):
                                   c = p.line(x=df_pred.index, line_width = 2.5, y=df_pred[station_i], alpha=0.6, line_dash = 'dashed',color = Plasma256[int(k*255/len(station))])
                                   legend_it.append((f'Prediction_{station_i}_q{q_i}', [c]))
                     else :
                            c = p.line(x=df_prediction.index, line_width = 2.5, y=df_prediction[station_i], alpha=0.6, line_dash = 'dashed',color = Plasma256[int(k*255/len(station))])
                            legend_it.append((f'Prediction_{station_i}', [c]))

       # Add rugby matches :
       for kick_time in kick_off_time:
              box = BoxAnnotation(left=kick_time - timedelta(minutes=1) , right=kick_time+ timedelta(minutes=1) ,
                                   fill_alpha=0.3, fill_color='darkgray')
              p.add_layout(box)
              # Ajouter une box verticale pour la période de +/- 'range'
              box = BoxAnnotation(left=kick_time - timedelta(minutes=range), right=kick_time + timedelta(minutes=range),
                                   fill_alpha=0.3, fill_color='lightgray')
              p.add_layout(box)

     
       p.xaxis.major_label_orientation = 1.2  # Pour faire pivoter les labels des x
       p.xaxis.formatter=DatetimeTickFormatter(
            months="%b",
            days="%a %d %b",
            hours="%a %d %b %H:%M",
            minutes="%a %d  %H:%M"
                )
       legend = Legend(items=legend_it)
       legend.click_policy="hide"
       p.add_layout(legend, 'right')

       if bool_show:
              output_notebook()
              show(p)

       return p

def plot_prediction_error(df_true,df_prediction,station,metrics =['mae','mse','mape'], title = '',width=1500,height=400,bool_show=False, min_flow = 20):
       legend_it = []
       p = figure(x_axis_type="datetime", title= title,
                     width=width,height=height)
       if type(station) != list:
             station = [station]
       
       for ind_station,station_i in enumerate(station):
              for k,metric in enumerate(metrics):
                     error = error_along_ts(predict= df_prediction[station_i],real= df_true[station_i],metric = metric,min_flow=min_flow,normalize=True)
                     df_error = pd.DataFrame(error.numpy(), index = df_true.index, columns = [station_i])
                     
                     c = p.line(x=df_error.index, line_width = 2.5, y=df_error[station_i], alpha=0.8,color = Set3_12[k+2])
                     legend_it.append((f"{metric}_{station_i}", [c]))


       p.xaxis.major_label_orientation = 1.2  # Pour faire pivoter les labels des x
       legend = Legend(items=legend_it)
       legend.click_policy="hide"
       p.add_layout(legend, 'right')

       if bool_show:
              output_notebook()
              show(p)

       return p

def str_valeur(valeur):
    ''' 
    Return an adapted str format for visualisation. 
    '''
    if valeur < 1:
        return("{:.2e}".format(valeur))  
    else:
        return("{:.2f}".format(valeur).rstrip('0').rstrip('.'))


def plot_loss_from_trainer(trainer,width=400,height=1500,bool_show=False):
       p = figure(title='Training and Validation loss',
                     width=width,height=height)
       legend_it = []
       colors = ['blue','red']

       if len(trainer.train_loss) > 0:
              for k,training_mode in enumerate(['train','valid']):
                     name= f"{training_mode}_loss"
                     loss_list = getattr(trainer,name)
                     c = p.line(x=np.arange(len(loss_list)), y=loss_list, alpha=0.8,color = colors[k])

                     last_loss = str_valeur(loss_list[-1])
                     best_loss = str_valeur(np.min(np.array(loss_list)))
                     displayed_legend = f"{name} \n   Last loss: {last_loss} \n   Best loss: {best_loss}"
                     
                     if training_mode == 'valid':
                            if 'MPIW' in list(trainer.performance['test_metrics'].keys()):
                                   test_mpiw = str_valeur(trainer.performance['test_metrics']['MPIW'])
                                   test_picp =  "{:.2%}".format(trainer.performance['test_metrics']['PICP'])
                                   displayed_legend = f"{displayed_legend}\n   Test MPIW: {test_mpiw}\n   Test PICP: {test_picp}"
                            if 'mse' in list(trainer.performance['test_metrics'].keys()):
                                   test_mae = str_valeur(trainer.performance['test_metrics']['mae'])
                                   test_mse = str_valeur(trainer.performance['test_metrics']['mse'])
                                   displayed_legend = f"{displayed_legend}\n   Test MSE: {test_mse}\n   Test MAE: {test_mae}"

                     legend_it.append((displayed_legend, [c]))

              p.xaxis.major_label_orientation = 1.2  # Pour faire pivoter les labels des x
              legend = Legend(items=legend_it)
              p.add_layout(legend, 'below')

       if bool_show:
              output_notebook()
              show(p)

       return p


def plot_TS(list_df_ts,width=400,height=1500,bool_show=False,title=f"Time Serie Intensity of NetMob apps consumption",scatter = False,x_datetime = True):
       if x_datetime:
             p = figure(x_axis_type="datetime",title=title,
                            width=width,height=height)
       else:
             p = figure(title=title,
                            width=width,height=height)
       legend_it = []
       colors = palette

       if not(type(list_df_ts)==list):
              list_df_ts= [list_df_ts]
       
       nb_cols = [df_i.shape[1] for df_i in list_df_ts]
       total_nb_ts = int(np.sum(np.array(nb_cols)))
       for i,df_i in enumerate(list_df_ts):
              for k,column in enumerate(df_i.columns):
                     col_ind = np.sum(np.array(nb_cols[:i]))+k
                     if scatter: 
                            c = p.scatter(x=df_i.index, y=df_i[column], alpha=0.8,color = colors[int(col_ind*(255/total_nb_ts))])
                     else:
                            c = p.line(x=df_i.index, y=df_i[column], alpha=0.8,color = colors[int(col_ind*(255/total_nb_ts))])
                     displayed_legend = str(column)
                     legend_it.append((displayed_legend, [c]))

       p.xaxis.major_label_orientation = 1.2  # Pour faire pivoter les labels des x
       legend = Legend(items=legend_it)
       legend.click_policy="hide"
       p.add_layout(legend, 'right')
       if x_datetime:
              p.xaxis.formatter=DatetimeTickFormatter(
              months="%b",
              days="%a %d %b",
              hours="%a %d %b %H:%M",
              minutes="%a %d  %H:%M"
                     )

       if bool_show:
              output_notebook()
              show(p)

       return p