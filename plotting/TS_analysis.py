from bokeh.plotting import figure
from bokeh.models import Legend,RangeTool
from bokeh.plotting import figure
from bokeh.models import Legend
from bokeh.models import BoxAnnotation
import torch 
from datetime import timedelta
import pandas as pd 
from bokeh.palettes import Set3_12 as palette
from bokeh.plotting import output_notebook

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

def drag_selection_box(df,p1,p2=None,width=1500, height=150):

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
    return(select)




def plot_single_point_prediction(df_true,df_prediction,station,title = '',kick_off_time = [], range = None,width=1500,height=400,show=False):
       legend_it = []
       p = figure(x_axis_type="datetime", title= title,
                     width=1500,height=400)

       c = p.line(x=df_true.index, line_width = 2.5, y=df_true[station], alpha=0.8,  legend_label = f'{station}',color = 'blue')
       legend_it.append(('True', [c]))

       c = p.line(x=df_prediction.index, line_width = 2.5, y=df_prediction[station], alpha=0.8,  legend_label = f'{station}',color = 'red')
       legend_it.append(('Prediction', [c]))


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
       legend = Legend(items=legend_it)
       p.add_layout(legend, 'right')

       if show:
              output_notebook()
              show(p)

       return p

def plot_prediction_error(df_true,df_prediction,station,metrics =['mae','mse','mape'], title = '',width=1500,height=400,show=False, min_flow = 20):
       legend_it = []
       p = figure(x_axis_type="datetime", title= title,
                     width=1500,height=400)
       
       def f_error(predict,real,metric):
              real = torch.tensor(real).reshape(-1)
              predict = torch.tensor(predict).reshape(-1)

              mask = real>min_flow
              error = torch.full(real.shape, -1.0)  # Remplir avec -1 par défaut
              if metric == 'mape':
                     error[mask] = 100 * (torch.abs(real[mask] - predict[mask]) / real[mask]) 

              elif metric == 'mae':
                     err = torch.abs(real[mask] - predict[mask])
                     error[mask] = 100 * err/err.max()
              elif metric == 'mse':
                     err = (real[mask] - predict[mask])**2
                     error[mask] = 100 * err/err.max()
              else:
                     raise NotImplementedError
              
              return(error)
       
       for k,metric in enumerate(metrics):
              error = f_error(predict= df_prediction[station],real= df_true[station],metric = metric)
              df_error = pd.DataFrame(error.numpy(), index = df_true.index, columns = [station])
              
              c = p.line(x=df_error.index, line_width = 2.5, y=df_error[station], alpha=0.8,color = palette[k+2])
              legend_it.append((metric, [c]))

       p.xaxis.major_label_orientation = 1.2  # Pour faire pivoter les labels des x
       legend = Legend(items=legend_it)
       legend.click_policy="hide"
       p.add_layout(legend, 'right')

       if show:
              output_notebook()
              show(p)

       return p