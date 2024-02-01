import pandas as pd
from utilities import str_xa2int, str2neg
from datetime import datetime,timedelta

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
