import torch
import os 
import sys 
import itertools

current_file_path = os.path.abspath(os.path.dirname(__file__))
working_dir = os.path.abspath(os.path.join(current_file_path,'..'))

if working_dir not in sys.path:
    sys.path.insert(0, working_dir)

from build_netmob_data import load_subway_shp,load_netmob_gdf,find_ids_within_epsilon

if torch.cuda.is_available():
    data_folder_path = '../../../../data/'
    netmob_data_folder_path = f"{data_folder_path}NetMob/"
    PATH_iris = f'{data_folder_path}lyon_iris_shapefile/'
else:
    data_folder_path = '../../../data/'
    netmob_data_folder_path = f"{data_folder_path}NetMob/"
    PATH_iris = '../../../Data/lyon_iris_shapefile/'

def buffer_between_tile_ids_and_subway_station(epsilon,netmob_data_folder_path,PATH_iris):
    ''' 
    args
    ------
    epsilon : radius, in meter 
    
    '''
    # Load Ref Subway: 
    ref_subway = load_subway_shp(folder_path = data_folder_path)

    # Load subway gdf adn NetMob gdf
    Netmob_gdf,_ = load_netmob_gdf(folder_path = netmob_data_folder_path,
                                data_folder = PATH_iris, 
                                geojson_path = 'Lyon.geojson',
                                zones_path = 'lyon.shp')
    Netmob_gdf_dropped = Netmob_gdf.drop_duplicates(subset = ['tile_id'])  # Some Doubles are exis

    # Get Cell-Id within epsilon : 
    result,joined = find_ids_within_epsilon(Netmob_gdf_dropped,ref_subway,epsilon=epsilon) 
    Netmob_gdf_dropped = Netmob_gdf_dropped.set_index('tile_id')

    maxi_nb_tile =  result.apply(lambda row: len(row.tile_id),axis=1).max()
    print(f"Maximum number of NetMob Cell associated to a subway station: {maxi_nb_tile}")

    tile_ids = list(set(itertools.chain.from_iterable(result.tile_id)))

    gdf_stations = Netmob_gdf_dropped.loc[tile_ids]
    return(gdf_stations,joined,result,maxi_nb_tile)

if __name__ == '__main__':

    # from tile_ids_around_stations import buffer_between_tile_ids_and_subway_station,netmob_data_folder_path,PATH_iris

    epsilon = 300
    (gdf_stations,joined,result,maxi_nb_tile) = buffer_between_tile_ids_and_subway_station(epsilon,netmob_data_folder_path,PATH_iris)