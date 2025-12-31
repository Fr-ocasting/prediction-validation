from bokeh.plotting import figure
from bokeh.models import Legend,BoxAnnotation,DatetimeTickFormatter,RangeTool,Band,ColumnDataSource,LegendItem,CustomJS,FixedTicker
import torch 
from datetime import timedelta
import datetime
import pandas as pd 
from bokeh.palettes import Set3_12,Dark2
from bokeh.palettes import Plasma256 
from bokeh.palettes import Turbo256 as palette
from bokeh.models.widgets import Div
from bokeh.layouts import column
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
    
from pipeline.utils.metrics import error_along_ts


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



def plot_line_and_buffer(mean_df, median_df, std_df,title, columns = None ,colors = None, width=800, height=400, legend_str = None,fill_alpha =0.3,title_font_size='16pt'):
    
    dummy_date = datetime.date(2025, 1, 1)
    mean_df_internal = mean_df.copy()
    std_df_internal = std_df.copy()
    
    mean_df_internal.index = [datetime.datetime.combine(dummy_date, i) for i in mean_df_internal.index]
    std_df_internal.index = [datetime.datetime.combine(dummy_date, i) for i in std_df_internal.index]
    
    p = figure(title=title, width=width, height=height, x_axis_type="datetime")
    use_palette = False
    legend_items = []
    if colors is None:
        colors = palette
        use_palette = True
    if columns is None:
        columns = mean_df_internal.columns
    nb_cols = len(columns)
    for k, column in enumerate(columns):
        if use_palette:
            pos = int(k * (255 / nb_cols))
        else:
            pos = k
        color_i = colors[pos]

        line_renderer = p.line(x=mean_df_internal.index, y=mean_df_internal[column], alpha=0.8, color=color_i)

        dict_source = {
            'time': std_df_internal.index,
            'lower': mean_df_internal[column].values - std_df_internal[column].values,
            'upper': mean_df_internal[column].values + std_df_internal[column].values
        }
        source_interval = ColumnDataSource(pd.DataFrame(dict_source))
        band = Band(base="time", lower="lower", upper="upper", source=source_interval,
                    fill_alpha=fill_alpha, fill_color=color_i, line_width=0)
        p.add_layout(band)

        callback = CustomJS(args={'band': band}, code="band.visible = this.visible;")
        line_renderer.js_on_change('visible', callback)

        band_proxy_renderer = p.patch([], [], fill_color=color_i, fill_alpha=fill_alpha, line_width=0)
        
        if legend_str is None:
              legend_label = str(column)
        else:
              legend_label = legend_str[k]
        
        legend_item = LegendItem(label=legend_label, renderers=[line_renderer, band_proxy_renderer])
        legend_items.append(legend_item)

    # 1. On sélectionne les graduations souhaitées sous forme de DatetimeIndex
    desired_ticks_dt = mean_df_internal.index[::2]
    
    # 2. CORRECTION FINALE : On convertit ces dates en millisecondes pour le FixedTicker
    ticks_in_ms = (desired_ticks_dt.astype('int64') // 10**6).tolist()
    
    # 3. On passe la liste de nombres (millisecondes) au FixedTicker
    p.xaxis.ticker = FixedTicker(ticks=ticks_in_ms)
    
    # Le formatter saura interpréter ces millisecondes comme des dates et les afficher en HH:MM
    p.xaxis.formatter = DatetimeTickFormatter(hours=["%H:%M"], days=["%H:%M"], months=["%H:%M"], years=["%H:%M"])

    p.xaxis.major_label_orientation = 1.2
    
    legend = Legend(items=legend_items) # , location="center"
    
    legend.click_policy = "hide"
#     p.add_layout(legend)
    p.add_layout(legend, 'right')
       # change title font size:

    p.title.text_font_size = title_font_size
    
    output_notebook()
    show(p)

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


def plot_single_point_prediction(df_true,df_prediction,station,title = '',kick_off_time = [], range = None,width=1500,height=400,bool_show=False,out_dim_factor=1,nb_step_ahead=1,horizon_step=1,freq='6min'):
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


       for k,spatial_unit_i in enumerate(df_true.columns):
              c = p.line(x=df_true.index, line_width = 2.5, y=df_true[spatial_unit_i], alpha=0.8,color = Plasma256[int(k*255/len(station))])
              legend_it.append((f'True_{spatial_unit_i}', [c]))

       if df_prediction is None:
              df_prediction = []
       if not type(df_prediction) == list:
              df_prediction = [df_prediction]

       if 'min' in freq:
             freq_num = int(freq.replace('min','')) 
             freq_label = 'min'
       elif ('h' in freq) or ('H' in freq):
             freq = freq.lower()
             freq_num = int(freq.replace('h','')) 
             freq_label = 'h'
       elif 'd' in freq:
             freq_num = int(freq.replace('D',''))
             freq_label = 'D'
       else:
             raise ValueError(f"Frequency {freq} not recognized. Use 'min', 'h', or 'd'.")
       

       for df_prediction_i in df_prediction:
              for column_i in df_prediction_i.columns:
                     station_i = column_i[0]
                     horizon = int(column_i[1][1:])*horizon_step
                     q_i = int(column_i[2][1:])
                     ind_station = station.index(station_i)

                     # Draw Line:
                     c = p.line(x=df_prediction_i.index, 
                            line_width = 2.5, 
                            y=df_prediction_i[column_i], 
                            alpha=0.8*(1-horizon/nb_step_ahead/2), 
                            line_dash = 'dashed',
                            color = Plasma256[int(ind_station*255/len(station))])

                     # Add legend

                     legend_str = f'Pred_{station_i}_{horizon*freq_num}{freq_label}'
                     if out_dim_factor>1: legend_str = f"{legend_str}_q{q_i}"
                     legend_it.append((legend_str, [c]))

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
                     displayed_legend = f"{name} loss: best (last) {best_loss} ({last_loss})"
                     legend_it.append((displayed_legend, [c]))

              p.xaxis.major_label_orientation = 1.2  # Pour faire pivoter les labels des x
              legend = Legend(items=legend_it)
              p.add_layout(legend, 'below')

              if trainer.args.loss_function_type in ['MSE','masked_mae','masked_mse','HuberLoss','huber_loss','masked_huber_loss']:
                     test_mae = ' // '.join([str_valeur(trainer.performance['test_metrics'][f'mae_h{h}']) for h in range(trainer.args.horizon_step,trainer.args.step_ahead+1,trainer.args.horizon_step)])
                     test_mse =' // '.join([str_valeur(trainer.performance['test_metrics'][f'mse_h{h}']) for h in range(trainer.args.horizon_step,trainer.args.step_ahead+1,trainer.args.horizon_step)])
                     text = Div(text = f"<b>Test MAE:</b> {test_mae} <br> <b>Test MSE:</b> {test_mse}", width=width, height=height//3)
              elif trainer.args.loss_function_type == 'quantile':
                     test_mpiw = str_valeur(trainer.performance['test_metrics']['MPIW'])
                     test_picp =  "{:.2%}".format(trainer.performance['test_metrics']['PICP'])
                     text = Div(text = f"<b>Test MPIW:</b> {test_mpiw} <br> <b>Test PICP:</b> {test_picp}", width=width, height=height//3)
              else:
                    raise NotImplementedError(f"Loss function {trainer.args.loss_function_type} not implemented for plotting")
              layout = column(p, text)
       else:
             return None

       if bool_show:
              output_notebook()
              show(layout)

       return layout


def plot_TS(list_df_ts,width=1500,height=400,bool_show=False,title=f"Time Serie Intensity of NetMob apps consumption",scatter = False,x_datetime = True, 
            only_hours = False,
            std_band = None,
              x_axis_label = None,
              y_axis_label = None,
            ):
       if x_datetime:
             p = figure(x_axis_type="datetime",title=title,
                            width=width,height=height)
       else:
             p = figure(title=title,
                            width=width,height=height)
       legend_it = []



       if not(type(list_df_ts)==list):
              list_df_ts= [list_df_ts]
              list_std_band = [std_band]
       
       nb_cols = [df_i.shape[1] for df_i in list_df_ts]
       total_nb_ts = int(np.sum(np.array(nb_cols)))
       col_ind = 0
       for i,df_i in enumerate(list_df_ts):
              for k,column in enumerate(df_i.columns):
                     if total_nb_ts > 12:
                            col_ind = np.sum(np.array(nb_cols[:i]))+k
                            color_i = palette[int(col_ind*(255/total_nb_ts))]
                     elif total_nb_ts > 8:
                           color_i = Set3_12[col_ind]
                           col_ind = col_ind+1
                     else:
                           color_i = Dark2[8][col_ind]
                           col_ind = col_ind+1

                           
                     if scatter: 
                            c = p.scatter(x=df_i.index, y=df_i[column], alpha=0.8,color = color_i)
                     else:
                            c = p.line(x=df_i.index, y=df_i[column], alpha=0.8,color = color_i)
                     displayed_legend = str(column)
                     legend_it.append((displayed_legend, [c]))

                     if std_band is not None:
                            std_i = list_std_band[i][column]
                            dict_source = {
                                 'time': df_i.index,
                                 'lower': df_i[column].values - std_i.values,
                                 'upper': df_i[column].values + std_i.values
                             }
                            source_interval = ColumnDataSource(pd.DataFrame(dict_source))
                            band = Band(base="time", lower="lower", upper="upper", source=source_interval,
                                         fill_alpha=0.3, fill_color=color_i, line_width=0)
                            p.add_layout(band)
              

       p.xaxis.major_label_orientation = 1.2  # Pour faire pivoter les labels des x
       legend = Legend(items=legend_it)
       legend.click_policy="hide"
       p.add_layout(legend, 'right')
       if x_datetime:
              if only_hours:
                     p.xaxis.formatter=DatetimeTickFormatter(
                          hours="%H:%M",
                          days="%H:%M",
                          months="%H:%M",
                          years="%H:%M"
                                 )
              else:
                     p.xaxis.formatter=DatetimeTickFormatter(
                     months="%b",
                     days="%a %d %b",
                     hours="%a %d %b %H:%M",
                     minutes="%a %d  %H:%M"
                            )
       # set x axis label
       if x_axis_label is not None:
              p.xaxis.axis_label = x_axis_label
       else:    
              if x_datetime:
                     p.xaxis.axis_label = "Time"
              else:
                     p.xaxis.axis_label = "Index"
       # set y axis label
       if y_axis_label is not None: 
              p.yaxis.axis_label = y_axis_label
       else:
             p.yaxis.axis_label = "Intensity"

       if bool_show:
              output_notebook()
              show(p)




       return p