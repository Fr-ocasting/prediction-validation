import os 
import glob 
import pandas as pd
FOLDER_PATH = '../../../../../../../data/rrochas/raw_data/Manhattan/citibike-tripdata'
SAVE_PATH = '../../../../../../../data/rrochas/prediction_validation/Manhattan'

## Open city_bike datasets
for year in range(2019, 2024):  # 2019 - 2024
    for month in range(1, 13):
        print(f"Loading {year} {month}...")
        bike_save_path  = f'{SAVE_PATH}/city_bike_{year}'
        if not os.path.exists(bike_save_path):
            os.mkdir(bike_save_path)
        csv_month_year = glob.glob(os.path.join(FOLDER_PATH,f'{year}-citibike-tripdata/{year}{month:02d}-citibike-tripdata/{year}{month:02d}-citibike-tripdata_*.csv'))
        globals()[f'city_bike_{year}_{month}'] = pd.DataFrame()
        for csv_path in csv_month_year:
            city_bike= pd.read_csv(csv_path,index_col=0)
            globals()[f'city_bike_{year}_{month}'] = pd.concat([globals()[f'city_bike_{year}_{month}'],city_bike])

        ## Aggregate city_bike data every X minutes:
        for freq in ['15min', '30min', '1h']:
            if 'started_at' in city_bike.columns:
                key_started_at = 'started_at'
                key_ended_at = 'ended_at'
                key_start_station_id = 'start_station_id'
                key_end_station_id = 'end_station_id'
            else:
                key_started_at = 'starttime'
                key_ended_at = 'stoptime'
                key_start_station_id = 'start station id'
                key_end_station_id = 'end station id'

            globals()[f'city_bike_{year}_{month}'][key_started_at] = pd.to_datetime(globals()[f'city_bike_{year}_{month}'][key_started_at])
            globals()[f'city_bike_{year}_{month}'][key_ended_at] = pd.to_datetime(globals()[f'city_bike_{year}_{month}'][key_ended_at])
            globals()[f'city_bike_{year}_{month}_emitted_freq{freq}'] = globals()[f'city_bike_{year}_{month}'].groupby([key_start_station_id,pd.Grouper(key=key_started_at, freq=freq)]).agg(Flow = (key_start_station_id, 'count'))
            globals()[f'city_bike_{year}_{month}_attracted_freq{freq}'] = globals()[f'city_bike_{year}_{month}'].groupby([key_end_station_id,pd.Grouper(key=key_ended_at, freq=freq)]).agg(Flow = (key_ended_at, 'count'))
            globals()[f'city_bike_{year}_{month}_OD_freq{freq}'] = globals()[f'city_bike_{year}_{month}'].groupby([key_start_station_id,key_end_station_id,pd.Grouper(key=key_started_at, freq=freq)]).agg(Flow = (key_started_at, 'count'))



            globals()[f'city_bike_{year}_{month}_emitted_freq{freq}'].to_csv(f'{bike_save_path}/{month:02d}_{freq}_emitted.csv')
            globals()[f'city_bike_{year}_{month}_attracted_freq{freq}'].to_csv(f'{bike_save_path}/{month:02d}_{freq}_attracted.csv')
            globals()[f'city_bike_{year}_{month}_OD_freq{freq}'].to_csv(f'{bike_save_path}/{month:02d}_{freq}_OD.csv')

            print(f"Saved {bike_save_path}/{month:02d}_{freq}_emitted.csv")
            print(f"Saved {bike_save_path}/{month:02d}_{freq}_attracted.csv")
            print(f"Saved {bike_save_path}/{month:02d}_{freq}_OD.csv")


# if __name__ == "__main__":
#     # Load city bike data: 
#     year = 2021
#     month = 1
#     bike_save_path  = f'{SAVE_PATH}/city_bike_{year}'
#     df_test = pd.read_csv(f'{bike_save_path}/{month:02d}_15min_emitted.csv',index_col=0,dtype={0: str} ) 