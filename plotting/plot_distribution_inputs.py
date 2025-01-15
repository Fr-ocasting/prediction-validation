import itertools
from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Whisker
import pandas as pd
import numpy as np 
from bokeh.plotting import output_notebook

def plot_scatter_distribution_by_calendar_group(list_of_point_to_plots,x_ticks,station_name):
    data = {'x': [], 'values': []}

    for i, points in enumerate(list_of_point_to_plots):
        data['x'].extend([x_ticks[i]] * len(points))
        data['values'].extend(points)

    # Source des données
    source = ColumnDataSource(pd.DataFrame(data))

    # Scatter plot
    scatter_fig = figure(x_range=x_ticks, title=f"Scatter Plot of Distributions for station {station_name}", height=300, width=1600)
    scatter_fig.scatter(x='x', y='values', source=source, size=5, alpha=0.6)
    scatter_fig.xaxis.major_label_orientation = 0.8

    return scatter_fig


def plot_distribution_by_calendar_group(list_of_point_to_plots,x_ticks,station_name):
    boxplot_data = {'x': [], 'q1': [], 'q2': [], 'q3': [], 'lower': [], 'upper': []}

    for i, points in enumerate(list_of_point_to_plots):
        if len(points) == 0:
            # Ajouter un espace vide pour préserver l'espacement
            boxplot_data['x'].append(x_ticks[i])
            boxplot_data['q1'].append(0)
            boxplot_data['q2'].append(0)
            boxplot_data['q3'].append(0)
            boxplot_data['lower'].append(0)
            boxplot_data['upper'].append(0)
            continue

        q1 = np.percentile(points, 25) 
        q2 = np.percentile(points, 50)  # Médiane
        q3 = np.percentile(points, 75) 
        iqr = q3 - q1
        lower_bound = max(q1 - 1.5 * iqr, np.min(points)) 
        upper_bound = min(q3 + 1.5 * iqr, np.max(points)) 

        boxplot_data['x'].append(x_ticks[i])
        boxplot_data['q1'].append(q1)
        boxplot_data['q2'].append(q2)
        boxplot_data['q3'].append(q3)
        boxplot_data['lower'].append(lower_bound)
        boxplot_data['upper'].append(upper_bound)

    source_box = ColumnDataSource(boxplot_data)

    # Créer la figure pour le boxplot
    box_fig = figure(x_range=x_ticks, title=f"Box Plot of Distributions for station {station_name}", height=300, width=1600)
    box_fig.segment(x0='x', y0='upper', x1='x', y1='q3', source=source_box, line_width=2, color="black")
    box_fig.segment(x0='x', y0='lower', x1='x', y1='q1', source=source_box, line_width=2, color="black")

    # Box Plot
    box_fig.vbar(x='x', width=0.7, top='q3', bottom='q1', source=source_box, fill_color="skyblue", line_color="black")

    # Add maedian
    box_fig.segment(x0='x', y0='q2', x1='x', y1='q2', source=source_box, line_width=2, color="black")

    # Box plot 
    box_fig.add_layout(Whisker(source=source_box, base='x', upper='upper', lower='lower'))
    box_fig.xaxis.major_label_orientation = 0.8
    return box_fig

def get_usefull_params(ds,t_minus_1,agg,posible_weekdays,posible_hours,posible_minutes,s_weekdays,s_hours,s_minutes):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    x_ticks = []
    if agg is None : 
        tuples = list(itertools.product(posible_weekdays,posible_hours,posible_minutes))
        for weekday,hour,minutes in tuples:
            x_ticks.append(f"{hour}:{minutes}h - {days[weekday]}")
        
        def add_to_point_to_plots(tuples,station_ind,s_weekdays=s_weekdays,s_hours=s_hours,s_minutes=s_minutes):
            list_of_point_to_plots = []
            for weekday,hour,minute in tuples:
                indexes = t_minus_1[s_hours.isin([hour])&s_weekdays.isin([weekday])&s_minutes.isin([minute])].index
                clustered_volume = ds.U_train[indexes,station_ind,-1].numpy()
                list_of_point_to_plots.append(clustered_volume)
            return list_of_point_to_plots


    elif agg == 'weekday_hour': 
        tuples = list(itertools.product(posible_weekdays,posible_hours))
        for weekday,hour in tuples:
            x_ticks.append(f"{hour}h - {days[weekday]}")

        def add_to_point_to_plots(tuples,station_ind,s_weekdays=s_weekdays,s_hours=s_hours,s_minutes=s_minutes):
            list_of_point_to_plots = []
            for weekday,hour in tuples:
                indexes = t_minus_1[s_hours.isin([hour])&s_weekdays.isin([weekday])].index
                clustered_volume = ds.U_train[indexes,station_ind,-1].numpy()
                list_of_point_to_plots.append(clustered_volume)
            return list_of_point_to_plots
    
    elif agg == 'hour':
        tuples = posible_hours
        for hour in tuples:
            x_ticks.append(f"{hour}h")

        def add_to_point_to_plots(tuples,station_ind,s_weekdays=s_weekdays,s_hours=s_hours,s_minutes=s_minutes):
            list_of_point_to_plots = []
            for hour in tuples:
                indexes = t_minus_1[s_hours.isin([hour])].index
                clustered_volume = ds.U_train[indexes,station_ind,-1].numpy()
                list_of_point_to_plots.append(clustered_volume)
            return list_of_point_to_plots
    else: 
        raise NotImplementedError
    
    return add_to_point_to_plots,tuples,x_ticks