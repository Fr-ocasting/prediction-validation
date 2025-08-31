import os 
import sys

# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
import numpy as np
import pandas as pd

# df: shape (7392, 40), index = DatetimeIndex freq=15min
# ts_stations_epsilons_transfers_apps: shape (40, 7392, CPi)

def group_by_weekday_hour_minutes(lag_corrs,station_i,lag_k):
    df_corr_i_k = lag_corrs[station_i][f'lag_{lag_k}'].copy()
    df_corr_i_k['weekday'] = df_corr_i_k.index.weekday
    df_corr_i_k['hour'] = df_corr_i_k.index.hour
    df_corr_i_k['minute'] = df_corr_i_k.index.minute
    df_corr_i_k = df_corr_i_k.groupby(['weekday','hour','minute']).mean()
    return df_corr_i_k


def lag_correlations(df, arr,lags=[0,1,2,3], window=6,name_conextual= None):
    '''
    Return a dictionnary of lag correlation between time-series.
    Some NaN value can appear if one of the time-serie is constant through a full window (i.e during night time for Subway-In)

    df : subway dataset with N time-series, one per stations
    arr : np.array shape [N,T,P], P time-serie for each of the N stations
    
    output: 
    lag_corrs: dict.   -> {station_0: {'lag_0': DataFrame, 
                                       'lag_1': DataFrame, 
                                        ... 
                                        },
                           ...
                           station_i:
                        }
    >>> lag_corrs[i]['lag_2'] => DataFrame rolling correlation pour station i au lag 2
    >>> lag_corrs[i]['lag_k'][t]: correspond to the correlation between the contextual series at the station i and subway-in at the station i on the window [t-window+1, t-window+2,...,t]
    and considering the lag (time shift) t.
    >>> The first Non-null correlation is from the indice window-1 (results[station_i][f'lag_{lag_k}'][window-1:])

    '''
    results = {}

    for i in range(arr.shape[0]):
        ref_series = df.iloc[:, i]
        comp_array = arr[i]  # shape (temps_filtré, 6)
        ref_df = pd.DataFrame(ref_series)
        corr_dict = {}
        for lag in lags:
            # Décalage de la série de référence
            shifted_ref = ref_df.shift(lag).dropna()
            
            # Rolling correlation sur chaque colonne du tableau comp_array
            corrs = []
            for contextual_ind in range(comp_array.shape[1]):
                if lag == 0:
                    shifted_contextual_i = comp_array[:, contextual_ind]
                else:
                    shifted_contextual_i = comp_array[:-lag, contextual_ind]
                tmp = pd.DataFrame({
                    "ref": shifted_ref.values.reshape(-1),
                    "comp": pd.Series(shifted_contextual_i) #, index=ref_series.index
                })
                rolling_corr = tmp["ref"].rolling(window).corr(tmp["comp"])
                corrs.append(rolling_corr)
            df_corr = pd.concat(corrs, axis=1)
            df_corr.index = ref_series.index[lag:]
            if name_conextual is not None:
                df_corr.columns = [name_conextual[j] for j in range(comp_array.shape[1])]
            corr_dict[f'lag_{lag}'] = df_corr
        results[i] = corr_dict
    return results