import sys
import os
import pandas as pd
import geopandas as gpd

current_file_path = os.path.abspath(os.getcwd())
parent_dir = os.path.abspath(os.path.join(current_file_path, '..','..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from constants.config import local_get_args
from constants.paths import FOLDER_PATH
from shapely.geometry import Point
from shapely.geometry import LineString
from load_inputs.Lyon.pt.subway_in import get_trigram_correspondance



def load_dataset(signal):
    if signal == 'subway_in' : 
        from load_inputs.Lyon.pt.subway_in import load_data, START, END,FILE_NAME
    if signal == 'subway_out' :
        from load_inputs.Lyon.pt.subway_out import load_data,START, END,FILE_NAME


    invalid_dates = []
    minmaxnorm = True
    standardize = False  # Set to True if you want to standardize the data
    config = {'model_name': 'STGCN',
            'freq' : '15min',
            'dataset_names':[signal],
            'dataset_for_coverage' : [signal],
            'target_data': signal,
            'contextual_kwargs':{signal:{},
                            },
            'step_ahead':1,
            'horizon_step':1,
            }

    args = local_get_args(config['model_name'],
                    args_init = None,
                    dataset_names=config['dataset_names'],
                    dataset_for_coverage=config['dataset_for_coverage'],
                    modification = config)


    coverage_period = pd.date_range(start=START, end=END, freq='15min')[:-1]  # Exclude the last date to match the data



    ds = load_data(FOLDER_PATH, 
                coverage_period=coverage_period, 
                invalid_dates=invalid_dates, 
                args=args, minmaxnorm=minmaxnorm,
                standardize=standardize, 
                normalize=True,
                filename=FILE_NAME,
                tensor_limits_keeper = None
                )
    df_raw = pd.DataFrame(ds.raw_values,columns = ds.spatial_unit, index = ds.df_dates['date'])
    df =  pd.DataFrame(ds.U_train[:,:,-1].detach().cpu().numpy(),columns = ds.spatial_unit, index = ds.tensor_limits_keeper.df_verif_train.iloc[:,-2] )
    return ds, df_raw, df



# Load Subway data
def load_subway_gdf(inflow,outflow,index_name):
    data_folder  = '../../../../../../data/rrochas/raw_data'
    subway_csv_path = 'keolis_data_2019-2020/ref_subway.csv'
    subway_lanes_path = 'keolis_data_2019-2020/metro_matching_TRI_FULL.CSV'
    df_subway = pd.read_csv(os.path.join(data_folder, subway_csv_path),index_col =0)
    subway_lanes = pd.read_csv(os.path.join(data_folder,subway_lanes_path),sep=';').rename(columns = {'LIGNE_A':'A', 'LIGNE_B':'B', 'LIGNE_C':'C', 'LIGNE_D':'D'})

    laneA =  ['PER','AMP','BEL', 'COR','HOT','FOC', 'MAS', 'CHA', 'REP','GRA','FLA','CUS', 'BON', 'SOI']
    laneB = ['CHA','BRO','PAR','GUI', 'SAX',  'MAC', 'JAU','DEB', 'GER','OGA' ]
    laneC = ['CUI', 'HEN','CRO','CPA','HOT']
    laneD = ['VAI','VMY','GOR','FOU', 'JEA','BEL','GIL','SAX','GAR','SAN', 'LUM','BLA', 'LAE', 'MER', 'PRY','VEN']

    df_subway_A = df_subway[df_subway['COD_TRG'].isin(laneA)].set_index('COD_TRG').reindex(laneA)
    df_subway_B = df_subway[df_subway['COD_TRG'].isin(laneB)].set_index('COD_TRG').reindex(laneB)
    df_subway_C = df_subway[df_subway['COD_TRG'].isin(laneC)].set_index('COD_TRG').reindex(laneC)
    df_subway_D = df_subway[df_subway['COD_TRG'].isin(laneD)].set_index('COD_TRG').reindex(laneD)
    df_correspondance = get_trigram_correspondance()

    gdf_subway_stations = gpd.GeoDataFrame()
    gdf_subway_lines = gpd.GeoDataFrame()
    for df_subway_i,lane_i in zip([df_subway_A,df_subway_B,df_subway_C,df_subway_D],['A','B','C','D']):
        df_subway_i.loc[:,'geometry'] = df_subway_i.apply(lambda row: Point(row['MEAN_X'],row['MEAN_Y']), axis=1)
        df_subway_i.loc[:,'lane'] = lane_i
        Linestring_i = LineString(df_subway_i['geometry'].tolist())
        gdf_subway_stations = pd.concat([gdf_subway_stations, gpd.GeoDataFrame(df_subway_i, geometry='geometry', crs='EPSG:4326')], ignore_index=True)
        gdf_subway_lines = pd.concat([gdf_subway_lines, gpd.GeoDataFrame({'lane':[lane_i],'geometry':[Linestring_i]}, geometry='geometry', crs='EPSG:4326')], ignore_index=True)

    gdf_subway_stations = gdf_subway_stations.merge(inflow.reset_index(),on = [index_name])
    gdf_subway_stations = gdf_subway_stations.merge(outflow.reset_index(),on = [index_name])
    return gdf_subway_stations, gdf_subway_lines, df_correspondance