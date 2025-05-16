import os 
import pandas as pd
import json
import geopandas as gpd
from shapely.geometry import Point


def load_gdf(save_folder):
    json_name = 'pvo_patrimoine_voirie.pvocomptagecriter.json'
    sensor_json = json.load(open(f"{save_folder}/{json_name}",'rb'))
    sensor_json['values']

    gdf_criter = gpd.GeoDataFrame(
        [
            {**feat,
            'geometry': Point(feat['lon'],feat['lat'])}
            for feat in sensor_json['values']
        ],
        crs="EPSG:4326"
    )
    return gdf_criter 



def load_csv_and_save(sensors,name,raw_data_folder_path,year,months,save_folder_path):
    df_all = pd.DataFrame()
    for nb_month in [3,4,5]:
        txt_path = f"{raw_data_folder_path}/6 min {year}/6mn_{str(nb_month).zfill(2)}_{months[nb_month-1]}_{year}.txt"

        df = pd.read_csv(txt_path, sep=';')
        format = "%d/%m/%Y %H:%M:%S"
        format_without_time = "%d/%m/%Y"
        format_hour = pd.to_datetime(df.HORODATE,format=format,errors = 'coerce')
        format_day = pd.to_datetime(df.HORODATE,format=format_without_time,errors = 'coerce')
        df.HORODATE = format_hour.combine_first(format_day)
        df['day'] = df.HORODATE.dt.day
        df['str_hour_min'] = df.HORODATE.dt.hour.astype(str) + pd.Series([':']*len(df)) + df.HORODATE.dt.minute.astype(str)
        df['hour_min'] = 10*df.HORODATE.dt.hour +  (df.HORODATE.dt.minute)/6
        df_filtered = df[df.ID_POINT_MESURE.isin(sensors)]
        print(f"On conserve seulement {'{:.2%}'.format(len(df_filtered)/len(df))} de la df initiale.")

        df_all = pd.concat([df_all,df_filtered])

    print('Couverture Temporelle: ',df_all.HORODATE.min(),df_all.HORODATE.max())
    df_all.drop(columns = ['NOM_POINT_MESURE','day','str_hour_min','hour_min'],inplace=True)
    #display(df_all.head())
    save_path = f"{save_folder_path}/{name}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    df_all.to_csv(f"{save_path}/{name}.csv")

    df_copy = df_all.copy()
    mask_ech_1_min_missing =  (df_copy['NOMBRE_ECH_1_MIN_MANQUANTS'] != 0)
    mask_flow_occupancy = (df_copy['TAUX_HEURE'] > 90) |  (df_copy['DEBIT_HEURE'] < 0) 
    df_copy = df_copy[~mask_flow_occupancy]

    print(f"proportion of record were TAUX_HEURE <= 90 and DEBIT_HEURE != -1 : {'{:.2%}'.format(len(df_copy)/len(df_all))}")
    for nb_min_manquante in [1,2,3,4,5]:
        print(f"proportion of record when {nb_min_manquante}/6 min are missing: {'{:.2%}'.format(len(df_copy[df_copy['NOMBRE_ECH_1_MIN_MANQUANTS'] >= nb_min_manquante])/len(df_copy))}")

    return df_all


if __name__ == "__main__":
    # Init
    root_repository = os.path.expanduser('~') #prediction_validation/
    raw_data_folder_path = f"{root_repository}/../../data/rrochas/raw_data/Comptages_Velo_Routier/CRITER"
    save_folder_path = f"{os.path.expanduser('~')}/../../data/rrochas/prediction_validation"
    year = 2019
    months = ['Janvier','Fevrier','Mars','Avril','Mai','Juin','Juillet','Aout','Septembre','Octobre','Novembre','Decembre']

    # Load GDF 
    gdf_criter = load_gdf(raw_data_folder_path)

    # Generate specific Graphs: 
    CRITER_3_4_5_lanes = gdf_criter[gdf_criter.nbvoies > 2]
    CRITER_3_4_5_lanes_sensor_id = CRITER_3_4_5_lanes['identifiantptm'].unique().astype(int)  # 158

    CRITER_sup_35000dailyflow = gdf_criter[(gdf_criter['moyennejoursouvrable'] > 35000)]
    CRITER_sup_35000dailyflow_sensor_id = CRITER_sup_35000dailyflow['identifiantptm'].unique().astype(int)  # 109

    CRITER_sup_35000dailyflow_3_4_5_lanes = gdf_criter[(gdf_criter['moyennejoursouvrable'] > 35000) & (gdf_criter['nbvoies'] > 2)]
    CRITER_sup_35000dailyflow_3_4_5_lanes_sensor_id  = CRITER_sup_35000dailyflow_3_4_5_lanes['identifiantptm'].unique().astype(int) #69

    CRITER_urban_between_15000_25000dailyflow = gdf_criter[(gdf_criter['moyennejoursouvrable'] > 15000) & (gdf_criter['moyennejoursouvrable'] < 25000) & (gdf_criter['nbvoies'] <= 2)]
    CRITER_urban_between_15000_25000dailyflow_sensor_id =  CRITER_urban_between_15000_25000dailyflow['identifiantptm'].unique().astype(int)


    sensor_list = [CRITER_3_4_5_lanes_sensor_id,CRITER_sup_35000dailyflow_sensor_id,CRITER_sup_35000dailyflow_3_4_5_lanes_sensor_id,CRITER_urban_between_15000_25000dailyflow_sensor_id]
    name_list = ['CRITER_3_4_5_lanes','CRITER_sup_35000dailyflow','CRITER_sup_35000dailyflow_3_4_5_lanes','CRITER_urban_between_15000_25000dailyflow']

    for sensors,name in zip(sensor_list,name_list):
        print('\n')
        print("Nb sensors: ",len(sensors))
        df_all = load_csv_and_save(sensors,name,raw_data_folder_path,year,months,save_folder_path)


        # Plot all sensors on folium map: 
        for gdf_i,sensors_ids in [CRITER_3_4_5_lanes,CRITER_sup_35000dailyflow,CRITER_sup_35000dailyflow_3_4_5_lanes,CRITER_urban_between_15000_25000dailyflow]:
            gdf_i.explore(tiles="Cartodb Positron")

