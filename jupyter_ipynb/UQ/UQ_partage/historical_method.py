import pandas as pd 
def get_stats_agg_per_calendar_group(train_df,method = 'week and hour',time_step_per_hour = 4):
    spatial_units = list(train_df.columns)
    train_df.index.name = 'datetime'
    train_df = train_df.reset_index()

    # Add columns weekday, hour, minutes : 
    if method == 'week and hour':
        train_df['weekday'] = train_df.datetime.dt.weekday
        train_df['hour'] = train_df.datetime.dt.hour
        groupby_col = ['weekday','hour']
        if time_step_per_hour > 1:
            train_df['minute'] = train_df.datetime.dt.minute
            groupby_col = groupby_col + ['minute']

    # Get 'mean' and 'std' for each spatial unit and temporal class (weekday,hour,min): 
    df_stats = train_df.groupby(groupby_col).agg({c : ['mean','std'] for c in spatial_units} )
    
    return(df_stats)
    
def get_upper_and_lower_bounds(df_stats,test_df,station_i):
    real = test_df.iloc[:,station_i]
    mean_i = df_stats.iloc[:,station_i]
    mean_i.name = 'mean'
    mean_i = mean_i.reset_index()

    std_i = df_stats.iloc[:,station_i+1]
    std_i.name = 'std'
    std_i = std_i.reset_index()

    datetime_index = real.index
    df_dates = pd.DataFrame(datetime_index,columns = ['datetime'])
    df_dates['hour'] = df_dates.datetime.dt.hour
    df_dates['weekday'] = df_dates.datetime.dt.weekday
    df_dates['minute'] = df_dates.datetime.dt.minute
    df_dates['weekday_hour_minute'] = df_dates['weekday'].astype(str) + '_' + df_dates['hour'].astype(str) + '_' + df_dates['minute'].astype(str)
    mean_i['weekday_hour_minute'] = mean_i['weekday'].astype(str) + '_' + mean_i['hour'].astype(str) + '_' + mean_i['minute'].astype(str)
    std_i['weekday_hour_minute'] = std_i['weekday'].astype(str) + '_' + std_i['hour'].astype(str) + '_' + std_i['minute'].astype(str)
    mean_i.drop(columns=['weekday', 'hour', 'minute'], inplace=True)
    std_i.drop(columns=['weekday', 'hour', 'minute'], inplace=True)
    df_dates.drop(columns=['weekday', 'hour', 'minute'], inplace=True)


    merged_df = df_dates.merge(mean_i.merge(std_i,on = 'weekday_hour_minute'), on='weekday_hour_minute')
    lower = merged_df['mean'] - merged_df['std']
    upper = merged_df['mean'] + merged_df['std']
    return real,lower,upper