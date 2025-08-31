import pandas as pd
from datetime import datetime
import numpy as np 
import geopandas as gpd
from shapely.geometry import Point 
from shapely.errors import ShapelyDeprecationWarning
import warnings

# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(current_file_path,'..'))
if ROOT not in sys.path:
    sys.path.insert(0,ROOT)
# ...

# Personnal import:
from pipeline.utils.utilities import get_distance_matrix
from pipeline.build_inputs.load_raw_data import load_subway_15_min 
from load_inputs.Lyon.pt.subway_in import get_trigram_correspondance

# ======================================================
# Function 
# ======================================================

def get_subway_lanes():
    subway_A = ['Perrache','Ampère Victor Hugo','Bellecour','Cordeliers', 
                'Hôtel de ville - Louis Pradel','Foch','Masséna','Charpennes', 
                'République Villeurbanne', 'Gratte Ciel','Flachet','Cusset','Laurent Bonnevay', 'La soie']
    
    subway_B = ['Charpennes','Brotteaux','Part-Dieu', 'Place Guichard', 'Saxe - Gambetta','Jean Macé',  'Place Jean Jaurès','Debourg', 'Stade de Gerland',"Gare d'Oullins"]

    subway_C = ['Hôtel de ville - Louis Pradel','Croix Paquet', 'Croix-Rousse', 'Hénon', 'Cuire']

    subway_D = ['Gare de Vaise','Valmy','Gorge de Loup', 'Vieux Lyon','Bellecour','Guillotière','Saxe - Gambetta','Garibaldi','Sans Souci', 'Monplaisir Lumière',  'Grange Blanche','Laënnec','Mermoz - Pinel','Parilly', 'Gare de Vénissieux'] 

    return({'A':subway_A,
            'B':subway_B,
            'C':subway_C,
            'D':subway_D})


def load_subway_shp(FOLDER_PATH,station_location_name):
    try:
        ref_subway = pd.read_csv(f'../{FOLDER_PATH}/{station_location_name}')[['MEAN_X','MEAN_Y','COD_TRG','LIB_STA_SIFO']]
    except:
        ref_subway = pd.read_csv(f'../{FOLDER_PATH}/{station_location_name}')[['lon','lat','COD_TRG','LIB_STA_SIFO']].rename(columns={'lon':'MEAN_X','lat':'MEAN_Y'})
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

    stations = dataset.spatial_unit

    subway_A = ['Perrache','Ampère Victor Hugo','Bellecour','Cordeliers', 
                'Hôtel de ville - Louis Pradel','Foch','Masséna','Charpennes', 
                'République Villeurbanne', 'Gratte Ciel','Flachet','Cusset','Laurent Bonnevay', 'La soie']
    
    subway_B = ['Charpennes','Brotteaux','Part-Dieu', 'Place Guichard', 'Saxe - Gambetta','Jean Macé',  'Place Jean Jaurès','Debourg', 'Stade de Gerland',"Gare d'Oullins"]

    subway_C = ['Hôtel de ville - Louis Pradel','Croix Paquet', 'Croix-Rousse', 'Hénon', 'Cuire']

    subway_D = ['Gare de Vaise','Valmy','Gorge de Loup', 'Vieux Lyon','Bellecour','Guillotière','Saxe - Gambetta','Garibaldi','Sans Souci', 'Monplaisir Lumière',  'Grange Blanche','Laënnec','Mermoz - Pinel','Parilly', 'Gare de Vénissieux']

    if type == 'adjacent':
        A = pd.DataFrame(0,index= stations,columns = stations)
        for i,station_i in enumerate(stations):
            for j, station_j in enumerate(stations[i:]):
                if ((i in subway_A) and (j in subway_A)) or ((i in subway_B) and (j in subway_B)) or ((i in subway_C) and (j in subway_C)) or ((i in subway_D) and (j in subway_D)):
                    A.loc[station_i,station_j] = 1
                    A.loc[station_j,station_i] = 1

        '''
        df_correspondance = get_trigram_correspondance()
        A = pd.DataFrame(0,index= stations,columns = stations)
        for lane in [subway_A,subway_B,subway_C,subway_D]:
            for i in range(len(lane)-1): 
                try:
                    A.loc[lane[i], lane[i+1]]
                    pos_i = lane[i]
                    pos_j = lane[i+1]
                except:
                    pos_i = df_correspondance.COD_TRG[df_correspondance.Station == lane[i]]
                    pos_j = df_correspondance.COD_TRG[df_correspondance.Station == lane[i+1]]

                A.loc[pos_i, pos_j] = 1
                A.loc[pos_j, pos_i] = 1  # Symmetry
        '''
        return(A)
    
    if type == 'correlation': 
        A_corr = pd.DataFrame(dataset.train_input).corr()  # Correlation only on the prior information, which mean only on the train dataset
        return(A_corr)
    
    if type == 'distance':
        # df_locations is supposed to contains elements of type 'shapely.geometry.Point' containing projected position in 'meter'
        df_locations = df_locations.to_crs('EPSG:2154')
        if ('AMP' in list(stations.values.reshape(-1))) and (not 'AMP' in df_locations.index):
            df_locations = df_locations.set_index('COD_TRG')
        else:
            raise NotImplementedError('Be carrefull. spatial unit within staions should match with index in df_locations')
        
        df_locations = df_locations.reindex(stations) # Be sure the order is the same 
        centroids = [[x,y] for x,y in zip(df_locations.geometry.x,df_locations.geometry.y)]
        A_dist = pd.DataFrame(get_distance_matrix(centroids,centroids, inv = True),index = stations, columns = stations)
        A_dist[A_dist< treshold] = 0

        return(A_dist)
                


def load_data_and_pivot(FOLDER_PATH, FILE_NAME, reindex,start=None, end = None):
    ''' Load 'subway_in' and 'subway_out' data. Re-organize them by Stations. 
    Values that doesn't exist for certain time-slot are set to 0

    ** Args :
        FOLDER_PATH = 'data/'
        FILE_NAME = 'Metro_15min_mar2019_mai2019.csv
        reindex: list of timestamp date coverage
    '''
    try :
        df_metro = pd.read_csv(f'{FOLDER_PATH}{FILE_NAME}',index_col = 0)
        df_metro.datetime = pd.to_datetime(df_metro.datetime)

    except:
        FOLDER_PATH = '../../../Data/keolis_data_2019-2020/'
        txt_path = "Métro 15 minutes 2019 2020.txt"
        df_metro_funi_2019_2020 = load_subway_15_min(FOLDER_PATH+txt_path)
        if start is not None:
            df_metro_funi_2019_2020 = df_metro_funi_2019_2020[(df_metro_funi_2019_2020.datetime >= start)&(df_metro_funi_2019_2020.datetime < end)]
        
    df_metro = df_metro_funi_2019_2020[df_metro_funi_2019_2020['Code ligne'].isin(['A','B','C','D'])]
    # For a same station, there are different sens. Let's aggregate them with 'sum' : 
    subway_in = pd.pivot_table(df_metro,index = 'datetime',columns = 'Station',values = 'in',aggfunc = 'sum', fill_value = 0).reindex(reindex).fillna(0)
    subway_out = pd.pivot_table(df_metro,index = 'datetime',columns = 'Station',values = 'out',aggfunc = 'sum', fill_value = 0).reindex(reindex).fillna(0)

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
    if False:
        # Init
        FOLDER_PATH = '../data'
        FILE_NAME = 'subway_IN_interpol_neg_15_min_2019_2020.csv'
        time_step_per_hour=4
        start,end = datetime(2019,3,16),datetime(2019,6,1)
        reindex = pd.date_range(start,end,freq = f'{60/time_step_per_hour}min')
        print(f'Number of time-slot: {4*24*(end-start).days}')

        # Load data
        subway_in,subway_out = load_data_and_pivot(FOLDER_PATH, FILE_NAME, reindex)

        # Pre-processing 
        subway_out = replace_negative(subway_out,method = 'linear')
        subway_in = replace_negative(subway_in,method = 'linear')


        # Set forbidden dates :
        # Data from  23_03_2019 14:00:00 to 28_04_2019 12:00:00 included should not been taken into account 
        invalid_dates = pd.date_range(datetime(2019,4,23,14),datetime(2019,4,28,14),freq = f'{60/time_step_per_hour}min')


    # Load Adj, Dist or Corr matrix : 
    from constants.paths import FOLDER_PATH
    from examples.benchmark import local_get_args,get_inputs


    station_location_name = 'ref_subway.csv'
    df_locations = load_subway_shp(FOLDER_PATH,station_location_name)

    dataset_names = ["subway_in"] # ["subway_in","calendar"] # ["subway_in"] # ['data_bidon']
    dataset_for_coverage = ['subway_in','netmob'] #  ['data_bidon','netmob'] #  ['subway_in','netmob'] 
    vision_model_name = None

    init_model_name ='STGCN' # start with # STGCN #CNN
    args= local_get_args(init_model_name,
                                                           dataset_names=dataset_names,
                                                           dataset_for_coverage=dataset_for_coverage,
                                                           modification = {})
    K_fold_splitter,K_subway_ds,dic_class2rpz = get_inputs(args,folds=[0])

    dataset = K_subway_ds[0]


    adj = load_adjacency_matrix(dataset, type = 'adjacent')
    corr =  load_adjacency_matrix(dataset, type = 'correlation')
    dist = load_adjacency_matrix(dataset, type = 'distance', df_locations = df_locations, treshold = 1e-4)

    if not os.path.exists(f'{ROOT}/{FOLDER_PATH}/{args.target_data}'):
        raise 'Folder not find'
    else:
        if not os.path.exists(f'{ROOT}/{FOLDER_PATH}/{args.target_data}/adj/'):
            os.mkdir(f'{ROOT}/{FOLDER_PATH}/{args.target_data}/adj/')
        adj.to_csv(f'{ROOT}/{FOLDER_PATH}/{args.target_data}/adj/adj.csv')
        corr.to_csv(f'{ROOT}/{FOLDER_PATH}/{args.target_data}/adj/corr.csv')
        dist.to_csv(f'{ROOT}/{FOLDER_PATH}/{args.target_data}/adj/dist.csv')
