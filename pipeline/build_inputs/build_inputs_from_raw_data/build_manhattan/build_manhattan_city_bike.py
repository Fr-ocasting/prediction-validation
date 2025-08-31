import os 
import glob 
import pandas as pd
FOLDER_PATH = '../../../../data/rrochas/raw_data/Manhattan/'
SAVE_PATH = '../../../../data/rrochas/prediction_validation/Manhattan'

## Open city_bike datasets
for year in range(2021, 2023):
    for month in range(1, 13):
        bike_save_path  = f'{SAVE_PATH}/city_bike_{year}'
        csv_month_year = glob.glob(os.path.join(FOLDER_PATH,f'{year}-citibike-tripdata/{year}{month:02d}-citibike-tripdata_*.csv'))
        globals()[f'city_bike_{year}_{month}'] = pd.DataFrame()
        for csv_path in csv_month_year:
            city_bike= pd.read_csv(csv_path,index_col=0)
            globals()[f'city_bike_{year}_{month}'] = pd.concat([globals()[f'city_bike_{year}_{month}'],city_bike])

        ## Aggregate city_bike data every X minutes:
        for freq in ['15min', '30min', '1h']:
            globals()[f'city_bike_{year}_{month}'].started_at = pd.to_datetime(globals()[f'city_bike_{year}_{month}'].started_at)
            globals()[f'city_bike_{year}_{month}'].ended_at = pd.to_datetime(globals()[f'city_bike_{year}_{month}'].ended_at)
            globals()[f'city_bike_{year}_{month}_emitted_freq{freq}'] = globals()[f'city_bike_{year}_{month}'].groupby(['start_station_id',pd.Grouper(key='started_at', freq=freq)]).agg(Flow = ('start_station_id', 'count'))
            globals()[f'city_bike_{year}_{month}_attracted_freq{freq}'] = globals()[f'city_bike_{year}_{month}'].groupby(['end_station_id',pd.Grouper(key='ended_at', freq=freq)]).agg(Flow = ('ended_at', 'count'))
            globals()[f'city_bike_{year}_{month}_OD_freq{freq}'] = globals()[f'city_bike_{year}_{month}'].groupby(['start_station_id','end_station_id',pd.Grouper(key='started_at', freq=freq)]).agg(Flow = ('started_at', 'count'))

            globals()[f'city_bike_{year}_{month}_emitted_freq{freq}'].to_csv(f'{bike_save_path}/{month:02d}_{freq}_emitted.csv')
            globals()[f'city_bike_{year}_{month}_attracted_freq{freq}'].to_csv(f'{bike_save_path}/{month:02d}_{freq}_attracted.csv')
            globals()[f'city_bike_{year}_{month}_OD_freq{freq}'].to_csv(f'{bike_save_path}/{month:02d}_{freq}_OD.csv')

            print(f"Saved {bike_save_path}/{month:02d}_{freq}_emitted.csv")
            print(f"Saved {bike_save_path}/{month:02d}_{freq}_attracted.csv")
            print(f"Saved {bike_save_path}/{month:02d}_{freq}_OD.csv")


if __name__ == "__main__":
    # Load city bike data: 
    year = 2021
    month = 1
    bike_save_path  = f'{SAVE_PATH}/city_bike_{year}'
    df_test = pd.read_csv(f'{bike_save_path}/{month:02d}_15min_emitted.csv',index_col=0,dtype={0: str} ) 