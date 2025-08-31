
import pandas as pd
import geopandas as gpd
from shapely import Point

FOLDER_PATH = '../../../../data/rrochas/raw_data/Manhattan'
SAVE_PATH = '../../../../data/rrochas/prediction_validation/Manhattan'

def transform_point(str_point):
    """Transform string 'Point <lat>, <lon>' to Shapely Point"""
    splitted = str_point.split(' ')
    lon = splitted[1].replace('(', '')
    lat = splitted[2].replace(')', '')
    return Point(float(lon), float(lat))


# Load NYC Subway Stations:
gdf = gpd.GeoDataFrame(pd.read_csv(f'{FOLDER_PATH}/MTA_Subway_Stations_20250812.csv',index_col=0).rename(columns={'Georeference': 'geometry'}))
gdf['geometry'] = gdf['geometry'].apply(transform_point)
gdf = gdf.set_geometry('geometry')
gdf.crs = 'EPSG:4326'  
print(len(gdf), 'stations in the initial GeoDataFrame')


# Load Manhattan Boundary :
manhattan_gdf = gpd.read_file(f'{FOLDER_PATH}/NYC_Borough_Boundary_6403672305752144374')
manhattan_gdf = manhattan_gdf[manhattan_gdf.BoroName == 'Manhattan']
manhattan_gdf = manhattan_gdf.to_crs('EPSG:4326')  # Convert initial 2232 to 4326

# Get intersection of subway stations with Manhattan boundary:
gdf = gdf[gdf.geometry.within(manhattan_gdf.geometry.values[0])]
print(len(gdf), 'stations in Manhattan GeoDataFrame')
gdf.explore()


if __name__ == "__main__":
    gdf = gpd.read_file(f'{SAVE_PATH}/subway_stations.shp',crs = 'EPSG:4326',index_col=0)