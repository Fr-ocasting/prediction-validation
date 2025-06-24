import sys 
import os 
import pandas as pd
import numpy as np
import torch 
from datetime import datetime
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from utils.utilities import restrain_df_to_specific_period
from build_inputs.load_preprocessed_dataset import load_input_and_preprocess
from typing import Tuple
from calendar_class import BANK_HOLIDAYS
from denoising.exponential import ExponentialSmoother

''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - add argument 'num_nodes', 'C' to the NameSpace. These are specific to this data
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''

DATA_SUBFOLDER = 'CRITER_3_4_5_lanes'
NAME = 'flow'
START = '03/01/2019'
END = '06/01/2019'
USELESS_DATES = {'hour':[], #[1,2,3,4,5,6],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }
FREQ = '6min'
list_of_invalid_period = []
list_of_invalid_period.append([datetime(2019,3,1,15,30),datetime(2019,3,1,15,42)])
list_of_invalid_period.append([datetime(2019,3,23,7,30),datetime(2019,3,23,9,0)])
list_of_invalid_period.append([datetime(2019,3,23,16,36),datetime(2019,3,23,20,12)])
list_of_invalid_period.append([datetime(2019,3,26,21,0),datetime(2019,3,26,21,12)])
list_of_invalid_period.append([datetime(2019,3,31,2,0),datetime(2019,3,31,3,0)])
C = 1
CITY = 'Lyon'
CHANNEL = 'DEBIT_HEURE'
NIGHT_HOUR = [0,1,2,3,4,5,6,21,22,23]


def load_data(FOLDER_PATH,invalid_dates,coverage_period,args,
             minmaxnorm,
              standardize,
              normalize= True,
              data_subfolder = DATA_SUBFOLDER,
              name=NAME,
              channel = CHANNEL,

              city = CITY,
                limit_max_nan=200,
                taux_heure_limit = 100,
                filter_high_outliers = True,
                fill_consecutive_nan_value_values = True,
                linear_interpolation_when_one_missing = True, 
                quantile = 0.99,
                accepting_factor = 1.5,
                consider_holidays = True, 
                agg_weekdays = True, 
                exponential_smoothing = True,
                alpha = 3.5,  

              ): # args,FOLDER_PATH,coverage_period = None
    assert '6min' == args.freq, f"Trying to apply a a {args.freq} temporal aggregation while CRITER is designed for 6min"

    df,df_anomalies_bool,df_mean,df_std,df_quantile_per_h,df_median_per_h,df_isnan = load_csvs(args,FOLDER_PATH,coverage_period,data_subfolder = data_subfolder,
                                                    channel = channel,
                                                    limit_max_nan = limit_max_nan,
                                                    taux_heure_limit = taux_heure_limit,
                                                    filter_high_outliers = filter_high_outliers,
                                                    fill_consecutive_nan_value_values = fill_consecutive_nan_value_values,
                                                    linear_interpolation_when_one_missing = linear_interpolation_when_one_missing, 
                                                    quantile =quantile,
                                                    accepting_factor = accepting_factor,
                                                    consider_holidays = consider_holidays, 
                                                    agg_weekdays = agg_weekdays, 
                                                    exponential_smoothing = exponential_smoothing,
                                                    alpha = alpha,  
                                                    )
    data_T = torch.tensor(df.values).float()
    dims = [0]# [0]  -> We are normalizing each time-serie independantly 


    processed_input = load_input_and_preprocess(dims = dims,normalize=normalize,invalid_dates=invalid_dates,args=args,data_T=data_T,coverage_period = coverage_period,name=f"{DATA_SUBFOLDER}_{name}",
                                                minmaxnorm=minmaxnorm,standardize=standardize) 

    # --- Finalisation Métadonnées ---
    processed_input.spatial_unit = df.columns.tolist()
    processed_input.C = C
    processed_input.adj_mx_path = f"{FOLDER_PATH}/{data_subfolder}/{data_subfolder}_adj.npy"
    processed_input.distance_mx_path = f"{FOLDER_PATH}/{data_subfolder}/{data_subfolder}_adj.npy"
    processed_input.raw_data_path =f"{FOLDER_PATH}/{data_subfolder}/{data_subfolder}.csv"
    processed_input.city = city
    return processed_input



def load_csvs(args,FOLDER_PATH,coverage_period,data_subfolder,channel,
              limit_max_nan:int,
              taux_heure_limit:int,
              filter_high_outliers: bool,
              fill_consecutive_nan_value_values: bool,
              linear_interpolation_when_one_missing: bool, 
              quantile : float,
              accepting_factor : float,
              consider_holidays : bool, 
              agg_weekdays : bool, 
              exponential_smoothing: bool,
              alpha : float, 

              ):
    """
    Load the data from the csv file and preprocess it.
    Args:
        args: Namespace, just here to access the frequency of the data
        FOLDER_PATH: path to the folder where the data is stored
        coverage_period: period of time to keep in the data
        data_subfolder: name of the subfolder where the data is stored
        channel: name of the channel to load
        limit_max_nan: maximum number of NaN values allowed in the data
        taux_heure_limit: maximum value of the occupancy rate allowed in the data
        filter_high_outliers: if True then filter high outliers
        fill_consecutive_nan_value_values: if True then fill consecutive NaN values with the median of the data
        linear_interpolation_when_one_missing: if True then fill NaN values with linear interpolation when only one is missing
        quantile: quantile to use for filtering high outliers
        accepting_factor: factor to use for filtering high outliers
        consider_holidays:  If True then remove holidays from weekdays to compute mean and std. Consider holidays as a sunday.
        agg_weekdays: If True then agg weekdays (i.e Monday to Friday) together before computing mean and std.
        exponential_smoothing: if True then apply exponential smoothing to the data
        alpha: Is considered as an anomaly if the value is outside [mean - alpha * std, mean + alpha * std]

    """

    # Load df: 
    df = pd.read_csv(f"{FOLDER_PATH}/{data_subfolder}/{data_subfolder}.csv",index_col = 0)
    df.HORODATE = pd.to_datetime(df.HORODATE)

    # Mask forbidden values: 
    mask = (df.TAUX_HEURE >= taux_heure_limit) | (df.DEBIT_HEURE < 0)
    df.loc[mask,['TAUX_HEURE','DEBIT_HEURE']] = np.nan

    # Pivot df : 
    pivoted_df = df.pivot_table(index = 'HORODATE',columns = 'ID_POINT_MESURE',values =channel).sort_index()
    reindex = pd.date_range(start = START,end =END,freq = FREQ)[:-1]
    invalid_dates = reindex.difference(pivoted_df.index)
    print(f"Number of invalid time-slots (i.e data when every single sensors does not have data): {len(invalid_dates)}")
    pivoted_df = pivoted_df.reindex(reindex)
    pivoted_df = restrain_df_to_specific_period(pivoted_df,coverage_period)

    df_isnan = pivoted_df.isna()

    # Fill values if expected during nightime (freeflow):
    df_filled = fill_nan_value_when_expected_freeflow(pivoted_df)

    # Get df_quantile_per_h:
    df_quantile_per_h,df_median_per_h = get_df_quantile(df_filled,quantile)

    # Remove Impossible outliers:
    if filter_high_outliers:
        df_filtered = _filter_high_outliers(df_filled,df_quantile_per_h,accepting_factor = accepting_factor)
    else:
        df_filtered = df_filled.copy()

    # Fill NaN value with linear interpolation when only one is missing:
    if linear_interpolation_when_one_missing: 
        df_interpolated = df_filtered.interpolate(method='linear', limit_direction='both',limit = 1)
    else:
        df_interpolated = df_filtered.copy()

    # Remove Sesor with too many NaN values:
    df_filtered_bis,sparse_columns = remove_sparse_sensor(df_interpolated,limit_max_nan+len(invalid_dates))
    
    # Resample if needed: 
    if args.freq != '6min':
        df_filtered_bis = df_filtered_bis.resample(args.freq).mean()

    print('Number of sensors after filter sparse sensor : ',len(df_filtered_bis.columns))
    
        
    # Re-compute df_quantile_per_h again
    df_quantile_per_h,df_median_per_h = get_df_quantile(df_filtered_bis,quantile)

    # Input NaN consecutive NaN value with historical regular behaviour or Median FreeFlow':
    if fill_consecutive_nan_value_values:
        inputed_df =  impute_scaled_median(df_filtered_bis,city = CITY, consider_holidays = consider_holidays, agg_weekdays=agg_weekdays) 
    else:
        inputed_df = df_filtered_bis.copy()
    # Exponential Filter: 
    if exponential_smoothing:
        expsmoother = ExponentialSmoother(alpha=0.3)
        df_smmoothed = pd.DataFrame(expsmoother(torch.tensor(inputed_df.values)),index = inputed_df.index,columns = inputed_df.columns)
    else:
        df_smmoothed = inputed_df.copy()
    # Identify anomalies and plot them 
    df_anomalies_bool,df_mean,df_std = create_df_bool_anomaly(df_smmoothed.copy(),alpha,consider_holidays,agg_weekdays,city = CITY)

    print(f" Data loaded with shape: {df_smmoothed.shape}")

    return df_smmoothed,df_anomalies_bool,df_mean,df_std,df_quantile_per_h,df_median_per_h,df_isnan



def remove_sparse_sensor(df,limit_max_nan = 200):
    # Remove Sesor with too many NaN values: 
    s_nb_nan_per_columns = df.copy().isna().sum()
    sparse_columns = s_nb_nan_per_columns[s_nb_nan_per_columns>limit_max_nan].index
    filtered_df = df.drop(columns = sparse_columns)

    # Input missing data by clusteing:

    print('nb sparse_columns : ',len(sparse_columns))

    return filtered_df,sparse_columns

def fill_nan_value_when_expected_freeflow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace each NaN in the DataFrame (only from 00:00 to 05:00) with the average
    recorded for the same day of the week, the same hour, and the same minute.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame indexed by a DatetimeIndex, with arbitrary columns (sensors).
    
    Returns:
    --------
    pd.DataFrame
        Copy of the DataFrame with NaNs potentially filled for hours 00:00-05:00.
    """
    print('number of nan values before filling : ',df.isna().sum().sum())
    # Mask for night time : from midnight to 6 am  (excluded)
    night_mask = df.index.hour.isin(range(6))

    # Compute grouped mean (weekday, hour, minute):
    means = df.groupby([df.index.weekday, df.index.hour, df.index.minute]).transform('mean')

    # Mask of (time-slot, sensor) to fill : NaN and night time
    mask_night_and_NaN = night_mask[:, None] & df.isna()

    # Fill Values only if NaN within the mask 
    df_filled = df.where(~mask_night_and_NaN,means).copy()

    return df_filled





def get_df_correspondance(df,agg_weekdays):
    df_correspondance = pd.DataFrame(index=df.index.copy())
    if not agg_weekdays:
        df_correspondance["weekday"] = df_correspondance.index.weekday          # 0 = Monday … 6 = Sunday
    df_correspondance["hour"]   = df_correspondance.index.hour
    df_correspondance["minute"] = df_correspondance.index.minute
    df_correspondance = df_correspondance.reset_index()  
    return df_correspondance

def get_df_mean_std(df: pd.DataFrame, consider_holidays = True,agg_weekdays = True,city = 'Lyon') -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_correspondance = get_df_correspondance(df,agg_weekdays)

    if consider_holidays:
        df_is_bankholiday = df[[]].copy()
        df_is_bankholiday['is_bankholiday'] = pd.Series(df_is_bankholiday.index.strftime("%Y-%m-%d"),index = df_is_bankholiday.index).apply(lambda date : date in BANK_HOLIDAYS[city])
        # calcul des statistiques pour jours réguliers (hors fériés et dimanches)
        if agg_weekdays:
            mask_reg = ~(df_is_bankholiday['is_bankholiday'] | (df_is_bankholiday.index.weekday == 6) | (df_is_bankholiday.index.weekday == 5))
            df_reg = df[mask_reg].copy()
            df_reg['hour'] = df_reg.index.hour  
            df_reg['minute'] = df_reg.index.minute
            stats_mean_reg = df_reg.groupby(['hour', 'minute']).mean().reset_index()
            stats_std_reg = df_reg.groupby(['hour', 'minute']).std().reset_index()
        else:
            mask_reg = ~(df_is_bankholiday['is_bankholiday'] | (df_is_bankholiday.index.weekday == 6))
            df_reg = df[mask_reg].copy()
            df_reg['weekday'] = df_reg.index.weekday
            df_reg['hour'] = df_reg.index.hour  
            df_reg['minute'] = df_reg.index.minute
            stats_mean_reg = df_reg.groupby(['weekday', 'hour', 'minute']).mean().reset_index()
            stats_std_reg = df_reg.groupby(['weekday', 'hour', 'minute']).std().reset_index()

        # calcul des statistiques pour dimanches et jours fériés
        if agg_weekdays:
            mask_hol = df_is_bankholiday['is_bankholiday'] | (df_is_bankholiday.index.weekday == 6)| (df_is_bankholiday.index.weekday == 5)
            df_hol  = df[mask_hol].copy()
            df_hol['hour'] = df_hol.index.hour  
            df_hol['minute'] = df_hol.index.minute
            stats_mean_hol = df_hol.groupby(['hour', 'minute']).mean().reset_index()
            stats_std_hol =  df_hol.groupby(['hour', 'minute']).std().reset_index()
        else:
            mask_hol = df_is_bankholiday['is_bankholiday'] | (df_is_bankholiday.index.weekday == 6) 
            df_hol  = df[mask_hol].copy()
            df_hol['weekday'] = df_hol.index.weekday
            df_hol['hour'] = df_hol.index.hour  
            df_hol['minute'] = df_hol.index.minute
            stats_mean_hol = df_hol.groupby(['hour', 'minute']).mean().reset_index()
            stats_std_hol =  df_hol.groupby(['hour', 'minute']).std().reset_index()


        # fusion pour jours réguliers
        if agg_weekdays:
            df_mean_reg = df_correspondance.merge(stats_mean_reg, on=['hour','minute'], how='left').drop(columns=['hour','minute']).set_index('index')
            df_std_reg = df_correspondance.merge(stats_std_reg, on=['hour','minute'], how='left').drop(columns=['hour','minute']).set_index('index')
        else:
            df_mean_reg = df_correspondance.merge(stats_mean_reg, on=['weekday','hour','minute'], how='left').drop(columns=['weekday','hour','minute']).set_index('index')
            df_std_reg = df_correspondance.merge(stats_std_reg, on=['weekday','hour','minute'], how='left').drop(columns=['weekday','hour','minute']).set_index('index')

        # fusion pour dimanches / fériés
        df_mean_hol = df_correspondance.merge(stats_mean_hol, on=['hour','minute'], how='left').drop(columns=['hour','minute']).set_index('index')
        df_std_hol = df_correspondance.merge(stats_std_hol, on=['hour','minute'], how='left').drop(columns=['hour','minute']).set_index('index')

        # construction des DataFrames finaux
        df_mean = df_mean_reg.copy()
        df_std  = df_std_reg.copy()
        df_mean.loc[mask_hol, :] = df_mean_hol.loc[mask_hol, :]
        df_std.loc[mask_hol, :]  = df_std_hol.loc[mask_hol, :]
    else:
        if agg_weekdays:
            stats_mean = df.groupby([df.index.hour, df.index.minute]).mean().sort_index().reset_index().rename(columns= {'level_0':'hour','level_1':'minute'})
            stats_std = df.groupby([df.index.hour, df.index.minute]).std().sort_index().reset_index().rename(columns= {'level_0':'hour','level_1':'minute'})

            df_mean = df_correspondance.merge(stats_mean,on=['hour','minute']).drop(columns=['hour','minute']).set_index('index')
            df_std = df_correspondance.merge(stats_std,on=['hour','minute']).drop(columns=['hour','minute']).set_index('index')
        else:
            stats_mean = df.groupby([df.index.weekday, df.index.hour, df.index.minute]).mean().sort_index().reset_index().rename(columns= {'level_0':'weekday','level_1':'hour','level_2':'minute'})
            stats_std = df.groupby([df.index.weekday, df.index.hour, df.index.minute]).std().sort_index().reset_index().rename(columns= {'level_0':'weekday','level_1':'hour','level_2':'minute'})

            df_mean = df_correspondance.merge(stats_mean,on=['weekday','hour','minute']).drop(columns=['weekday','hour','minute']).set_index('index')
            df_std = df_correspondance.merge(stats_std,on=['weekday','hour','minute']).drop(columns=['weekday','hour','minute']).set_index('index')

    return df_mean, df_std


def create_df_bool_anomaly(df: pd.DataFrame, alpha = 3,consider_holidays=True,agg_weekdays=False,city = 'Lyon') -> pd.DataFrame:
    """
    Identifies anomalies in a DataFrame based on historical data for the same time slot (weekday, hour, minute).
    An anomaly is a value outside the [mean - 3*std, mean + 3*std] interval.
    Optimized to avoid nested loops.

    Args:
        df (pd.DataFrame): DataFrame with time series data in columns and DateTimeIndex.

    Returns:
        pd.DataFrame: DataFrame of the same shape as df, with True where an anomaly is detected, False otherwise.
    """
    df_mean, df_std = get_df_mean_std(df, consider_holidays,agg_weekdays,city)
    df_anomalies_bool = (df < (df_mean - alpha * df_std)) | (df > (df_mean + alpha * df_std))
    print(f"\nTotal anomalies detected: {df_anomalies_bool.sum().sum()}")

    return df_anomalies_bool,df_mean,df_std


def get_df_quantile(filtered_df,quantile):
    df_copy = filtered_df.copy()
    df_copy['hour'] = df_copy.index.hour
    quantile_per_hour = df_copy.groupby(['hour']).quantile(quantile).reset_index()
    median_per_hour = df_copy.groupby(['hour']).median().reset_index()

    def get_df_from_agg_data(df_copy,quantile_per_hour):
        df_quantile_per_h = df_copy[[]].copy()
        df_quantile_per_h['hour'] = df_quantile_per_h.index.hour
        df_quantile_per_h = df_quantile_per_h.reset_index()    
        df_quantile_per_h = df_quantile_per_h.merge(quantile_per_hour, on=['hour'], how='left').drop(columns=['hour']).set_index('index')
        return df_quantile_per_h
    df_quantile_per_h = get_df_from_agg_data(df_copy,quantile_per_hour)
    df_median_per_h = get_df_from_agg_data(df_copy,median_per_hour)

    return df_quantile_per_h,df_median_per_h

def _filter_high_outliers(df,df_quantile_per_h,accepting_factor):
    mask = df > df_quantile_per_h * accepting_factor
    interpolated_df = df.copy()
    interpolated_df[mask] = np.nan  
    interpolated_df = interpolated_df.interpolate(method='linear', limit_direction='both')

    filtered_df = df* ~mask + interpolated_df * mask

    return filtered_df


# --------------------------------------------------------------------------- #
#  Helper: build the “day-type / hour / minute” template with MEDIANS
# --------------------------------------------------------------------------- #
def _get_df_median(df: pd.DataFrame,city: str = "Lyon",consider_holidays: bool = True,agg_weekdays: bool = True) -> pd.DataFrame:
    """
    Return, for every time-stamp in *df*, the median value observed in the past
    for the *same kind of day* (weekday vs. weekend / bank-holiday) **and**
    the *same (hour, minute)*.
    
    The column structure of the returned DataFrame is identical to *df*.
    """
    # ------------------------------------------------------------------ #
    #  1. Build helper DataFrame with the grouping keys
    # ------------------------------------------------------------------ #
    meta = get_df_correspondance(df,agg_weekdays)
    
    # ------------------------------------------------------------------ #
    #  2. Compute medians for every class
    # ------------------------------------------------------------------ #
    if consider_holidays:
        # Flag bank-holidays
        meta['is_bankholiday'] = meta['index'].apply(lambda date : date.strftime("%Y-%m-%d") in BANK_HOLIDAYS['Lyon'])
        # logical masks
        is_weekend = meta["index"].dt.weekday.isin([5, 6])  # Sat / Sun
        mask_reg   = ~(meta["is_bankholiday"] | is_weekend) # “regular” days
        mask_hol   = ~mask_reg                              # WE + holidays
    else:
        mask_reg = meta["index"].notna()  # everything
        mask_hol = ~mask_reg              # empty mask
    
    def _median_by_keys(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        return (
            frame.groupby(keys, dropna=False)       # keep NaNs out of the groups
            .median(numeric_only=True)
            .reset_index()
        )
    
    # ---------- Regular days ---------- #
    if agg_weekdays:
        keys_reg = ["hour", "minute"]
    else:
        keys_reg = ["weekday", "hour", "minute"]
    
    stats_median_reg = _median_by_keys(
        df.loc[mask_reg.values].assign(**{
            k: meta.loc[mask_reg, k].values for k in keys_reg
        }),
        keys_reg,
    )
    
    # ---------- Week-ends & holidays ---------- #
    stats_median_hol = _median_by_keys(
        df.loc[mask_hol.values].assign(
            hour   = meta.loc[mask_hol, "hour"].values,
            minute = meta.loc[mask_hol, "minute"].values,
        ),
        ["hour", "minute"],
    )
    
    # ------------------------------------------------------------------ #
    #  3. Re-attach those medians to the full time index
    # ------------------------------------------------------------------ #
    def _merge_template(stats: pd.DataFrame, on: list[str]) -> pd.DataFrame:
        out = meta.merge(stats, on=on, how="left")       # add median columns
        return out.set_index("index").drop(columns=on)
    
    df_median = _merge_template(stats_median_reg, keys_reg)
    if consider_holidays:
        # overwrite rows that belong to week-ends / holidays
        df_median.loc[mask_hol.values, :] = _merge_template(
            stats_median_hol, ["hour", "minute"]
        ).loc[mask_hol.values, :]
    
    return df_median,mask_reg,mask_hol,keys_reg
# --------------------------------------------------------------------------- #
#  Main function: impute with scaled medians
# --------------------------------------------------------------------------- #
def impute_scaled_median(df: pd.DataFrame,city: str = "Lyon",consider_holidays: bool = True,agg_weekdays: bool = True,) -> pd.DataFrame:
    """
    Replace NaNs in a time-series DataFrame *df* with the median value
    observed for the same day-type / hour / minute, **scaled** by an
    amplitude coefficient *α* that reflects whether that particular day is
    globally higher (>1) or lower (<1) than the historical median day.
    
    Parameters
    ----------
    df : pd.DataFrame
        DatetimeIndex (any frequency) and numeric columns to impute.
    bank_holidays : dict[str, Iterable[str]], optional
        Mapping “city → list of YYYY-MM-DD bank holidays”.
    city : str, default “Lyon”
        Key to pick the holiday list from *bank_holidays*.
    consider_holidays : bool, default True
        If True, week-ends and bank holidays share the same template.
    agg_weekdays : bool, default True
        If True, Monday→Friday are pooled together; else weekday is explicit.
    
    Returns
    -------
    pd.DataFrame
        Copy of *df* with NaNs replaced.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("`df` must have a DatetimeIndex.")
    
    df_out   = df.copy()
    df_med,mask_reg,mask_hol,keys_reg   = _get_df_median(
        df,
        city=city,
        consider_holidays=consider_holidays,
        agg_weekdays=agg_weekdays,
    )

    # ------------------------------------------------------------------ #
    #  2. Broadcast α to every time-stamp
    # ------------------------------------------------------------------ #
    df_reg = df_out.loc[mask_reg.values].copy()
    #df_reg['hour'] = df_reg.index.hour
    #df_reg['minute'] = df_reg.index.minute
    df_reg.loc[:, 'hour'] = df_reg.index.hour
    df_reg.loc[:, 'minute'] = df_reg.index.minute

    median_per_hour_minutes = df_reg.groupby(['hour','minute']).median()
    df_correspondance = get_df_correspondance(df_reg,True)
    df_expected_median_h_m = df_correspondance.merge(median_per_hour_minutes.reset_index(), on=['hour','minute'], how='left').drop(columns=['hour','minute']).set_index('index')

    df_hol = df_out.loc[mask_hol.values].copy()
    #df_hol['hour'] = df_hol.index.hour
    #df_hol['minute'] = df_hol.index.minute
    df_hol.loc[:, 'hour'] = df_hol.index.hour
    df_hol.loc[:, 'minute'] = df_hol.index.minute
    median_per_hour_minutes_hol = df_hol.groupby(['hour','minute']).median()
    df_correspondance_hol = get_df_correspondance(df_hol,True)
    df_expected_median_h_m_hol = df_correspondance_hol.merge(median_per_hour_minutes_hol.reset_index(), on=['hour','minute'], how='left').drop(columns=['hour','minute']).set_index('index')
    df_median_all = pd.concat([df_expected_median_h_m,df_expected_median_h_m_hol],axis=0).sort_index()

    alpha =  (df_out/df_median_all).copy()
    alpha[alpha.index.hour.isin(NIGHT_HOUR)] = 1

    df_coeff_by_date = alpha[~alpha.index.hour.isin(NIGHT_HOUR)].copy()
    df_coeff_by_date = df_coeff_by_date.groupby(df_coeff_by_date.index.date).mean()
    df_coeff_by_date.loc[:,'date'] = df_coeff_by_date.index

    alpha_reg =  alpha[~alpha.index.hour.isin(NIGHT_HOUR)][[]].copy()
    alpha_reg.loc[:,'date'] = alpha_reg.index.date
    alpha_reg.loc[:,'timestamp'] = alpha_reg.index
    alpha_reg = alpha_reg.merge(df_coeff_by_date, on='date').set_index('timestamp').drop(columns=['date'])
    alpha_reg
    alpha[~alpha.index.hour.isin(NIGHT_HOUR)] = alpha_reg
    # ------------------------------------------------------------------ #
    #  3. Imputation
    # ------------------------------------------------------------------ #
    nan_mask = df_out.isna()
    # scaled_median = median_template * α_day
    scaled_median  = df_med.mul(alpha, axis=0)
    df_out[nan_mask] = scaled_median[nan_mask]
    
    return df_out




# --------------------------------------------------------------------------- #
# TO REMOVE
# --------------------------------------------------------------------------- #

if False:
    def impute_isolated_anomalies_and_identify_chains(df: pd.DataFrame,
                                                df_bool_anomaly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Replaces isolated anomalies in df with linear interpolation and identifies non-isolated (chained) anomalies.
        Anomalies are first replaced by NaN in a copy of df to facilitate interpolation.

        Args:
            df (pd.DataFrame): DataFrame with time series data.
            df_bool_anomaly (pd.DataFrame): DataFrame indicating anomalies (True/False).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - df_imputed (pd.DataFrame): DataFrame with isolated anomalies imputed.
                - df_bool_chaining_anomaly (pd.DataFrame): DataFrame with True for chained anomalies.
        """
        df_imputed = df.copy()
        # Replace all identified anomalies with NaN to prepare for interpolation
        for col in df.columns:
            df_imputed.loc[df_bool_anomaly[col], col] = np.nan

        df_bool_chaining_anomaly = pd.DataFrame(False, index=df.index, columns=df.columns)

        for col in df.columns:
            anomalies_col = df_bool_anomaly[col]
            for i in range(len(anomalies_col)):
                if anomalies_col.iloc[i]:  # If it's an anomaly
                    is_isolated = True
                    is_chain_start_or_middle = False

                    # Check if it's part of a chain
                    if i > 0 and anomalies_col.iloc[i-1]: # Previous is anomaly
                        is_isolated = False
                        is_chain_start_or_middle = True
                    if i < len(anomalies_col) - 1 and anomalies_col.iloc[i+1]: # Next is anomaly
                        is_isolated = False
                        is_chain_start_or_middle = True

                    if is_chain_start_or_middle:
                        df_bool_chaining_anomaly.loc[anomalies_col.index[i], col] = True
                    elif is_isolated:
                        # Try to interpolate: df_imputed already has NaN at this anomaly.
                        # The .interpolate() method handles isolated NaNs if surrounded by valid numbers.
                        pass # Interpolation will be done column-wise later


            df_imputed[col] = df_imputed[col].interpolate(method='linear', limit_direction='both')
        print(f"\nTotal chained anomalies detected: {df_bool_chaining_anomaly.sum().sum()}")
        print(f"\nNaNs in original filtered_df['{loop_to_plot}']: {filtered_df[loop_to_plot].isna().sum()}")
        return df_imputed, df_bool_chaining_anomaly