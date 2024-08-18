import pandas as pd
from datetime import datetime,timedelta

# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal inputs:
from utils.utilities import str_xa2int, str2neg


def load_subway_15_min(txt_path, dates = None):
    ''' Return preprocessed df of "Métro 15 minutes". 
    dates : list of months, when dates[0] and dates[-1] are start and end of period. Must have '%MM-%YYYY' format
    '''
    df = pd.read_csv(txt_path, delimiter ="\t",low_memory=False).rename(columns = {' Date jour (CAS tr)':'date','Heure (CAS tr)':'hour','Nb entrées total (CAS tr)':'in','Nb sorties total (CAS tr)':'out'})

    # Tackle float values represented by str: 
    df['out'] = df['out'].transform(lambda x : str_xa2int(x))
    df['in'] = df['in'].transform(lambda x : str_xa2int(x))

    # Tackle String Values ('A', 'B'):
    df['in'] = df['in'].transform(lambda x : str2neg(x))  # Transform each string values like 'A', 'B', '1A'  and so on to '-1'. Keep NaN values as NaN values.
    df['out'] = df['out'].transform(lambda x : str2neg(x))  

    # Add '20' like '01_01_2020' instead of '01_01_20'
    df['date'] = df['date'].transform(lambda d : d+'20' if len(d)<10 else d)
    T_2020 = pd.to_datetime(df['date'] + ' ' + df['hour'], format='%d/%m/%Y %H:%M', errors='coerce')
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour'], format='%d/%m/%Y %H:%M:%S', errors='coerce').fillna(T_2020)

    # Keep usefull columns
    df = df[['datetime','Station','Code ligne','in','out']]

    # Restrain to usefull 'dates' : 
    if dates is not None: 
        start,end = datetime(int(dates[0].split('-')[1]),int(dates[0].split('-')[0]),1),datetime(int(dates[-1].split('-')[1]),int(dates[-1].split('-')[0])+1,1)
        df = df[(df.datetime >= start) & (df.datetime < end)]

    return(df)


def load_CRITER(txt_path):
    df = pd.read_csv(txt_path, sep=';',infer_datetime_format=True)
    format = "%d/%m/%Y %H:%M:%S"
    format_without_time = "%d/%m/%Y"
    format_hour = pd.to_datetime(df.HORODATE,format=format,errors = 'coerce')
    format_day = pd.to_datetime(df.HORODATE,format=format_without_time,errors = 'coerce')
    df.HORODATE = format_hour.combine_first(format_day)
    df['day'] = df.HORODATE.dt.day
    df['str_hour_min'] = df.HORODATE.dt.hour.astype(str) + pd.Series([':']*len(df)) + df.HORODATE.dt.minute.astype(str)
    df['hour_min'] = 10*df.HORODATE.dt.hour +  (df.HORODATE.dt.minute)/6
    return(df)

def load_netmob(txt_path,day):
    # let's make a list of 15 min time intervals to use as column names
    day_str = str(day)
    day = datetime.strptime(day_str, '%Y%m%d')
    times = [day + timedelta(minutes=15*i) for i in range(96)]
    times_str = [t.strftime('%H:%M') for t in times]

    # column names
    columns = ['tile_id'] + times_str

    df = pd.read_csv(txt_path, sep = ' ', names = columns).set_index(['tile_id'])
    return(df)