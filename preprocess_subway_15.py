import pandas as pd
from datetime import datetime

import numpy as np 

# Personnal Import 
from DL_utilities import DataSet
from load_data import load_subway_15_min 
from utilities import get_distance_matrix

import geopandas as gpd
from shapely.geometry import Point 
from shapely.errors import ShapelyDeprecationWarning
import warnings
# ======================================================
# Function 
# ======================================================

def get_trigram_correspondance():
    ''' Some surprise : 
        Vieux Lyon : Jea
        Gare d'oulins : OGA
    '''
    df = pd.DataFrame(columns = ['Station','COD_TRG'])
    df['COD_TRG'] = ['AMP','BEL','BRO','COR',
                     'CUI','CUS','FLA','GOR',
                     'BLA','GRA','GUI','GIL',
                     'HEN','HOT','LAE','MAS',
                     'MER','LUM','PRY','PER',
                     'SAN','SAX','VMY','JEA',
                     'BON','CHA','VAI','VEN',
                     'MAC','GAR','FOC','REP',
                     'GER','DEB','JAU','CPA',
                     'CRO','PAR','SOI','OGA']
    
    df['Station'] =['Ampère Victor Hugo','Bellecour','Brotteaux','Cordeliers',
                    'Cuire','Cusset','Flachet','Gorge de Loup',
                    'Grange Blanche','Gratte Ciel','Place Guichard','Guillotière',
                    'Hénon','Hôtel de ville - Louis Pradel','Laënnec','Masséna',
                    'Mermoz - Pinel','Monplaisir Lumière','Parilly','Perrache',
                    'Sans Souci','Saxe - Gambetta','Valmy','Vieux Lyon',
                    'Laurent Bonnevay','Charpennes','Gare de Vaise','Gare de Vénissieux',
                    'Jean Macé','Garibaldi','Foch','République Villeurbanne',
                    'Stade de Gerland','Debourg','Place Jean Jaurès','Croix Paquet',
                    'Croix-Rousse','Part-Dieu','La soie',"Gare d'Oullins"]
    return(df)


def load_subway_shp(folder_path,station_location_name):
    try:
        ref_subway = pd.read_csv(f'{folder_path}{station_location_name}')[['MEAN_X','MEAN_Y','COD_TRG','LIB_STA_SIFO']]
    except:
        ref_subway = pd.read_csv(f'{folder_path}{station_location_name}')[['lon','lat','COD_TRG','LIB_STA_SIFO']].rename(columns={'lon':'MEAN_X','lat':'MEAN_Y'})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ShapelyDeprecationWarning)
        ref_subway['geometry'] = ref_subway.apply(lambda row : Point(row.MEAN_X,row.MEAN_Y),axis = 1)
    ref_subway = gpd.GeoDataFrame(ref_subway)
    ref_subway = ref_subway[['COD_TRG','geometry']]
    ref_subway.crs = 'epsg:4326'

    df_correspondance = get_trigram_correspondance()
    ref_subway = ref_subway.merge(df_correspondance)
    ref_subway = ref_subway.set_index(['Station'])
    return(ref_subway)


def load_adjacency_matrix(dataset, type = 'adjacent', df_locations = None, treshold = 0):

    stations = dataset.columns

    subway_A = ['Perrache','Ampère Victor Hugo','Bellecour','Cordeliers', 
                'Hôtel de ville - Louis Pradel','Foch','Masséna','Charpennes', 
                'République Villeurbanne', 'Gratte Ciel','Flachet','Cusset','Laurent Bonnevay', 'La soie']
    
    subway_B = ['Charpennes','Brotteaux','Part-Dieu', 'Place Guichard', 'Saxe - Gambetta','Jean Macé',  'Place Jean Jaurès','Debourg', 'Stade de Gerland',"Gare d'Oullins"]

    subway_C = ['Hôtel de ville - Louis Pradel','Croix Paquet', 'Croix-Rousse', 'Hénon', 'Cuire']

    subway_D = ['Gare de Vaise','Valmy','Gorge de Loup', 'Vieux Lyon','Bellecour','Guillotière','Saxe - Gambetta','Garibaldi','Sans Souci', 'Monplaisir Lumière',  'Grange Blanche','Laënnec','Mermoz - Pinel','Parilly', 'Gare de Vénissieux']
      
    if type == 'adjacent':
        A = pd.DataFrame(0,index=  stations,columns = stations)
        for lane in [subway_A,subway_B,subway_C,subway_D]:
            for i in range(len(lane)-1): 
                A.loc[lane[i], lane[i+1]] = 1
                A.loc[lane[i+1], lane[i]] = 1  # Symmetry
        return(A)
    
    if type == 'correlation': 
        A_corr = dataset.df_train.corr()  # Correlation only on the prior information, which mean only on the train dataset
        return(A_corr)
    
    if type == 'distance':
        # df_locations is supposed to contains elements of type 'shapely.geometry.Point' containing projected position in 'meter'
        df_locations = df_locations.to_crs('EPSG:2154')
        df_locations = df_locations.reindex(stations) # Be sure the order is the same 
        centroids = [[x,y] for x,y in zip(df_locations.geometry.x,df_locations.geometry.y)]
        A_dist = pd.DataFrame(get_distance_matrix(centroids,centroids, inv = True),index = stations, columns = stations)
        A_dist[A_dist< treshold] = 0
        return(A_dist)
                


def load_data_and_pivot(folder_path, file_name, reindex):
    ''' Load 'subway_in' and 'subway_out' data. Re-organize them by Stations. 
    Values that doesn't exist for certain time-slot are set to 0

    ** Args :
        folder_path = 'data/'
        file_name = 'Metro_15min_mar2019_mai2019.csv
        reindex: list of timestamp date coverage
    '''
    try :
        df_metro = pd.read_csv(f'{folder_path}{file_name}',index_col = 0)
        df_metro.datetime = pd.to_datetime(df_metro.datetime)

    except:
        folder_path = '../../Data/keolis_data_2019-2020/'
        txt_path = "Métro 15 minutes 2019 2020.txt"
        df_metro_funi_2019_2020 = load_subway_15_min(folder_path+txt_path)
        df_metro = df_metro_funi_2019_2020[(df_metro_funi_2019_2020.datetime >= start)&(df_metro_funi_2019_2020.datetime < end)]

    # For a same station, there are different sens. Let's aggregate them with 'sum' : 
    subway_in = pd.pivot_table(df_metro,index = 'datetime',columns = 'Station',values = 'in',aggfunc = 'sum', fill_value = 0).reindex(reindex).fillna(0)
    subway_out = pd.pivot_table(df_metro,index = 'datetime',columns = 'Station',values = 'out',aggfunc = 'sum', fill_value = 0).reindex(reindex).fillna(0)

    subway_in = subway_in.drop(columns= ['Fourvière'])
    subway_out = subway_out.drop(columns= ['Fourvière'])

    return(subway_in,subway_out)


# Replace Negative values : 
def replace_negative(df,method = 'linear'):
    df = df.clip(lower=-1)
    df = df.replace(-1,np.nan)
    df = df.interpolate(method=method)
    return df


# ======================================================
# Application 
# ======================================================
if __name__ == '__main__':

    # Init
    folder_path = 'data/'
    file_name = 'Metro_15min_mar2019_mai2019.csv'
    time_step_per_hour=4
    H,W,D = 6,1,1
    step_ahead = 1
    train_prop = 0.6
    start,end = datetime(2019,3,16),datetime(2019,6,1)
    reindex = pd.date_range(start,end,freq = f'{60/time_step_per_hour}min')
    print(f'Number of time-slot: {4*24*(end-start).days}')

    # Load data
    subway_in,subway_out = load_data_and_pivot(folder_path, file_name, reindex)

    # Pre-processing 
    subway_out = replace_negative(subway_out,method = 'linear')
    subway_in = replace_negative(subway_in,method = 'linear')


    # Set forbidden dates :
    # Data from  23_03_2019 14:00:00 to 28_04_2019 12:00:00 included should not been taken into account 
    invalid_dates = pd.date_range(datetime(2019,4,23,14),datetime(2019,4,28,14),freq = f'{60/time_step_per_hour}min')