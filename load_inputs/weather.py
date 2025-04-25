import numpy as np
import pandas as pd
import bokeh as bk
from datetime import datetime

def date_transform(date):
    date = ''.join('-'.join('-'.join(date.split('/')).split(':')).split(' '))
    date = datetime.strptime(date, '%d-%m-%Y-%H')
    date = date.strftime('%Y-%m-%d-%H')
    return(date)


def load_weather_data(folder_path = 'Data/Meteo',
                     pathway = 'donnees-meteo.csv',
                     id_stations = [69029001,69299001],
                     columns = ['POSTE','DATE','T','RR1','DRR1','FF','GLO'],
                     dico = {'POSTE':'id_station','DATE':'date','T':'temperature', 
            'RR1':'precip','DRR1':'duree_prec','FF':'wind_ms','GLO':'solar'}
            ):
    # Get 'donnees-meteo' :
    df_weather = pd.read_csv(f"{folder_path}/{pathway}",on_bad_lines='skip', sep = ";",low_memory=False,)  # Opening data
    df_weather = df_weather[columns]  # Columns selections 
    df_weather = df_weather.rename(columns = dico)  # Rename 
    df_weather = df_weather[df_weather['id_station'].isin(id_stations)]  # Filter station
    
    df_weather['date'] = df_weather['date'].apply(lambda date : date_transform(date) )  # Date transform 
    df_weather['date'] = pd.to_datetime(df_weather['date'])

    return(df_weather)

def fill_empty_values(df_weather):
    grouped_df = df_weather.groupby('date')   # Group station
    for date,group in grouped_df:    
        for index, row in group.iterrows():
            for column in ['temperature','precip','duree_prec','wind_ms','solar']:
                if pd.isnull(row[column]):
                    df_weather.at[index, column] = group[group.index != index][column].values[0]  #Fill empty values 
    return(df_weather)

def sort_and_interpolate(df_weather):
    df_weather = df_weather.set_index(['id_station','date'])    # Sorr by id and by date
    df_weather = df_weather.sort_values(['id_station','date'])

    df_weather = df_weather.interpolate(method = 'linear',limit_direction='forward',axis = 0)   # Interpolate Empty values 
    
    # Convert french format (32,4) to US format (32.4)
    for column in ['temperature','precip','wind_ms']:
        df_weather[column] =  df_weather[column].apply(lambda t : float('.'.join(t.split(','))))
    return(df_weather)


if __name__ == "__main__":

    import os
    import sys 
    current_path = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.path.abspath(os.path.join(current_path, '..'))
    if working_dir not in sys.path:
        sys.path.insert(0, working_dir)

    from constants.paths import FOLDER_PATH

    folder_path = f"{FOLDER_PATH}/weather"
    csv_path = 'donnees-meteo.csv'
    id_stations = [69029001,69299001]  # Lyon Bron and Lyon St-exupery
    columns = ['POSTE','DATE','T','RR1','DRR1','FF','GLO']
    dico = {'POSTE':'id_station','DATE':'date','T':'temperature', 
            'RR1':'precip','DRR1':'duree_prec','FF':'wind_ms','GLO':'solar'}

    df_weather = load_weather_data(folder_path,csv_path,id_stations=id_stations,columns=columns,dico=dico)
    print(df_weather.head())