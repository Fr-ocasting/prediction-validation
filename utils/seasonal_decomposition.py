import sys 
import os 
import pandas as pd
import numpy as np
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

import pandas as pd
import numpy as np
import math 
from statsmodels.tsa.seasonal import seasonal_decompose, STL , MSTL
from constants.paths import USELESS_DATES
import matplotlib.pyplot as plt 

def fill_and_decompose_df(raw_values,df_verif_train,time_step_per_hour,columns,min_count = 10, periods = [24*7*4,24*4]):
    '''
    ds : PersonnalInput ou DataSet object 
    min_count : défini le minimum d'occurence d'un tuple (weekday,hour) pour qu'on y input avec la médiane.
                Sinon on input (weekday,hour) avec la mediane des valeurs aggrégée par (hour)
    period : liste de périodes associées aux différentes périodicités de la série temporelle. S
              Si on considère que y a D time-steps entre deux jours consécutif, et W entre deux semaines consécutives, alors period = [W,D] est adapté.
    '''
    dates_used_in_train = pd.Series(pd.concat([df_verif_train[c] for c in df_verif_train.columns]).unique()).sort_values() # Concat forbidden indexes within each columns

    # Get corresponding indices:
    df_verif_train_ind = pd.DataFrame()
    for c in df_verif_train.columns:
        df_verif_train_ind[c] = df_verif_train.index + int(c.split('t')[-1])
    ind_used_in_train = pd.Series(pd.concat([df_verif_train_ind[c] for c in df_verif_train_ind.columns]).unique()).sort_values()

    # Reindex 
    reindex_dates = pd.date_range(dates_used_in_train.min(),dates_used_in_train.max(),freq=f"{1/time_step_per_hour}h")
    reindex_dates = reindex_dates[~reindex_dates.hour.isin(USELESS_DATES['hour'])&~reindex_dates.hour.isin(USELESS_DATES['weekday'])]

    # Input Missing Values to get complete Time-Series
    time_series_for_seasonal_decomposition = impute_median_by_dow_hour(raw_values[ind_used_in_train].numpy(),dates_used_in_train,reindex_dates,columns,min_count)

   # MSTL on the filled Time-Series
    decomposition = {}
    for col in time_series_for_seasonal_decomposition.columns:
        decomposition_i = ts_decomposition(time_series_for_seasonal_decomposition[col], periods=periods,take_abs= False)
        decomposition[col] = decomposition_i

    return decomposition

def ts_decomposition(ts: pd.Series, periods: list = [7], take_abs: bool = False) -> pd.Series:
    """
    Décompose la série temporelle `ts` (index datetime) en (trend, season, resid).
    Retourne la composante de bruit (résidu) sous forme d'un pd.Series indexé
    par les mêmes datetimes."""
    if False:
        result = seasonal_decompose(ts, model='additive', period=period)
    if False : 
        stl = STL(ts, period=period)
        result = stl.fit()
    if True:
        mstl = MSTL(ts, periods=periods)
        result = mstl.fit()
    output  = {
        'trend': result.trend.dropna(),
        'seasonal': result.seasonal.dropna(),
        'resid': result.resid.dropna()  # Souvent NaN au debut ou fin 
    }

    # Si on veut juste l'amplitude du Bruit: 
    if take_abs:
        output['resid'] = output['resid'].abs()
    
    return output

def add_calendar_columns(df,calendar_informations):
    if 'weekday' in calendar_informations:
        df['weekday'] = df.index.weekday
    if 'hour' in calendar_informations:
        df['hour'] = df.index.hour
    if 'minute' in calendar_informations:
        df['minute'] = df.index.minute
    if 'hour_minute' in calendar_informations:
        df['hour_minute'] = df.apply(lambda row: f"{row['hour']}:{row['minute']}",axis=1)
    return df
 

def impute_median_by_dow_hour(data: pd.DataFrame,dates_used_in_train,reindex_dates,columns,min_count: int = 10) -> pd.DataFrame:
    """
    Impute les valeurs manquantes de `data` en se basant sur :
      1) La médiane par (jour_de_semaine, heure) si le nombre de points dans ce groupe >= min_count
      2) Sinon, la médiane par heure (tous jours confondus) en fallback.

    Hypothèse : df.index est de type DatetimeIndex
    """
    # Get time_serie_for_seasonal_decomposition
    df_base_for_inputation = pd.DataFrame(data = data,index = dates_used_in_train,columns=columns)
    columns = df_base_for_inputation.columns
    df_base_for_inputation = add_calendar_columns(df_base_for_inputation,['weekday','hour','minute'])

    # median by (weekday,hour,minute) 
    group_dow_hour_minute = df_base_for_inputation.groupby(['hour', 'weekday','minute'])
    median_dow_hour_minute = group_dow_hour_minute.transform('median')
    count_dow_hour_minute = group_dow_hour_minute.transform('count')

    # median by (weekday,hour) 
    group_dow_hour = df_base_for_inputation.groupby(['hour', 'weekday'])
    median_dow_hour = group_dow_hour.transform('median').drop(columns=['minute'])
    count_dow_hour = group_dow_hour.transform('count').drop(columns=['minute'])

    # median by (hour) 
    group_hour = df_base_for_inputation.groupby('hour')
    median_hour = group_hour.transform('median').drop(columns=['weekday','minute'])

    # Median by (weekday,hour,minute). If not enough then median by (weekday,hour). If not enough then median by (hour) 
    #impute_val = np.where(count_dow_hour < min_count, median_hour, median_dow_hour)
    impute_val1 = np.where(count_dow_hour_minute > min_count, median_dow_hour_minute, median_dow_hour)
    impute_val = np.where(count_dow_hour > min_count, impute_val1, median_hour)

    values_to_be_imputed = pd.DataFrame(impute_val,index = df_base_for_inputation.index,columns = columns)
    values_to_be_imputed = add_calendar_columns(values_to_be_imputed,['weekday','hour','minute','hour_minute'])


    # Re-index df:
    time_series_for_seasonal_decomposition = df_base_for_inputation.reindex(reindex_dates) 
    time_series_for_seasonal_decomposition = add_calendar_columns(time_series_for_seasonal_decomposition,['weekday','hour','minute','hour_minute'])

    # Extract Sub-df of Nan Values: 
    data_to_input = time_series_for_seasonal_decomposition[time_series_for_seasonal_decomposition.isna().sum(axis=1) > 0]

    #L_input_col = [values_to_be_imputed.pivot_table(index = 'hour',columns = 'weekday', values = col) for col in columns]
    L_input_col = [values_to_be_imputed.pivot_table(index = 'hour_minute',columns = 'weekday', values = col) for col in columns]
    data_to_input = data_to_input.apply(lambda row: replace_values(row,L_input_col,columns),axis = 1)
    time_series_for_seasonal_decomposition =time_series_for_seasonal_decomposition.fillna(data_to_input).drop(columns = ['weekday','hour','minute','hour_minute'])
    return time_series_for_seasonal_decomposition



def replace_values_by_colunm(row,col,values_to_be_imputed):
    if math.isnan(row[col]):
        row[col] = values_to_be_imputed.at[int(row['hour']),int(row['weekday'])]
    return row 


def replace_values(row,L_input_col,columns):
    for k,col in enumerate(columns):
        #row[col] = L_input_col[col].at[int(row['hour']),int(row['weekday'])]
        row[col] = L_input_col[k].at[row['hour_minute'],int(row['weekday'])]
    return row 


def recompose_ts_and_plot(noise_from_decomposition,name,start = 96*7, nb_timeslots= 96*2):
    df = pd.DataFrame({'trend': noise_from_decomposition['trend'],
                                    'resid': noise_from_decomposition['resid']})

    if type(noise_from_decomposition['seasonal']) == pd.DataFrame:
        for col_i in noise_from_decomposition['seasonal'].columns:
            df[col_i] = noise_from_decomposition['seasonal'][col_i]
    else:
        df['seasonal'] = noise_from_decomposition['seasonal']

    df['TS'] = df.sum(axis=1)
    df.iloc[start:start+nb_timeslots].plot()
    plt.title(f"Seasonal decomposition of {name}")