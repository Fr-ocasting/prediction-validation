import json 
import pandas as pd
import geopandas as gpd 
from os import listdir
import os 
import glob 
from shapely.geometry import Point,Polygon
from shapely.errors import ShapelyDeprecationWarning
import warnings 

def find_ids_within_epsilon(gdf1,gdf2,epsilon):
    gdf1 = gdf1.to_crs(epsg=2154)
    gdf2 = gdf2.to_crs(epsg=2154)

    gdf1['centroid'] = gdf1.geometry.centroid

    # Get Centroid
    centroids = gdf1[['tile_id', 'centroid']].copy()
    centroids = centroids.rename(columns={'centroid': 'geometry'})
    centroids = gpd.GeoDataFrame(centroids, geometry='geometry', crs=gdf1.crs)

    # Get buffer from 'Point'
    gdf2_buffered = gdf2.copy()
    gdf2_buffered['geometry'] = gdf2_buffered.geometry.buffer(epsilon)

    # Spatial Join: 
    joined = gpd.sjoin(centroids, gdf2_buffered, how='inner', predicate='intersects')

    result = joined.groupby('COD_TRG')['tile_id'].apply(list).reset_index()

    return result,joined

def load_subway_shp(folder_path = '../../Data/keolis_data_2019-2020/'):
    zones_shp_path = folder_path+'ref_subway.csv'

    ref_subway = pd.read_csv(zones_shp_path)[['MEAN_X','MEAN_Y','COD_TRG','LIB_STA_SIFO']]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ShapelyDeprecationWarning)
        ref_subway['geometry'] = ref_subway.apply(lambda row : Point(row.MEAN_X,row.MEAN_Y),axis = 1)
    ref_subway = gpd.GeoDataFrame(ref_subway)
    ref_subway = ref_subway[['COD_TRG','geometry']]
    ref_subway.crs = 'epsg:4326'

    return(ref_subway)




def load_netmob_json(data_folder, geojson_path = 'Lyon.geojson'):
    ''' Load GeoJson, and then the spatial correspondence '''
    Lyon = json.load(open(data_folder+geojson_path,'r'))
    Netmob_gdf = pd.json_normalize(Lyon, record_path =['features']).rename(columns = {'properties.tile_id' : 'tile_id', 'geometry.coordinates':'geometry'})[['tile_id','geometry']]
    Netmob_gdf.geometry = Netmob_gdf.geometry.transform(lambda x : Polygon(x[0]))
    
    city_dims = {'Lyon': (426, 287)}
    city_str = 'Lyon'
    n_rows, n_cols = city_dims[city_str]
    
    return(Netmob_gdf,n_rows,n_cols)

def restrain_netmob_to_Lyon(Netmob_gdf,folder_path,zones_path):
    ''' Restraint "Netmob_gdf" to the working area '''
    working_zones = gpd.read_file(f'{folder_path}{zones_path}')
    Netmob_gdf = gpd.GeoDataFrame(Netmob_gdf)
    Netmob_gdf.crs = 'epsg:4326'
    
    restrained_Lyon_gdf = gpd.sjoin(Netmob_gdf,working_zones,how = 'inner')
    return(restrained_Lyon_gdf,working_zones)


def load_netmob_restrained_to_lyon(folder_path,data_folder,
                    geojson_path = 'Lyon.geojson',
                    zones_path = 'lyon_iris_shapefile/lyon.shp',
                    ):

    Netmob_gdf,n_rows,n_cols = load_netmob_json(folder_path, geojson_path = geojson_path)
    Netmob_gdf['centroid_lonlat'] = Netmob_gdf.geometry.apply(lambda x : x.centroid)
    Netmob_gdf = Netmob_gdf.set_geometry('centroid_lonlat')
    Netmob_gdf_joined,working_zones = restrain_netmob_to_Lyon(Netmob_gdf,data_folder,zones_path) #Associate an square to an IRIS when the centroid is inside it
    
    return(Netmob_gdf_joined,working_zones)

def load_netmob_gdf(folder_path = '../../Data/NetMob/',data_folder = '../../Data/lyon_iris_shapefile/', geojson_path = 'Lyon.geojson',zones_path = 'lyon.shp'):
    Netmob_gdf_joined,working_zones = load_netmob_restrained_to_lyon(folder_path,data_folder,geojson_path,zones_path)
    Netmob_gdf_joined = Netmob_gdf_joined[['tile_id','INSEE_COM','NOM_COM','NOM_IRIS','geometry']]
    Netmob_gdf_joined = gpd.GeoDataFrame(Netmob_gdf_joined,crs='EPSG:4326') #Don't know why, but the geodataframe seem corrupted as we can't convert it into GeoJson, that's why we need to use "gpd.GeoDataFrame()"

    return(Netmob_gdf_joined)



def get_netmob_csv_paths(folder_path = '../../Data/NetMob/' ):
    apps = [app for app in listdir(folder_path) if ((app != 'Lyon.geojson') and (not app.startswith('.'))) ]   # Avoid hidden folder and Lyon.geojson
    folder_days = [[day for day in listdir(f'{folder_path}/{app}') if (not day.startswith('.')) ] for app in apps]
    day = folder_days[0]
    txt_paths = sorted(glob.glob(os.path.join(f'{folder_path}/{app}/{day}', "*.txt")))
