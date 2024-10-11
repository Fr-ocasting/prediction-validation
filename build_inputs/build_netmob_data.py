import json 
import pandas as pd
import geopandas as gpd 
from os import listdir
import os 
import glob 
from shapely.geometry import Point,Polygon
from shapely.errors import ShapelyDeprecationWarning
import warnings 
import torch
import torch.nn.functional as F
from datetime import datetime,timedelta
import numpy as np 



def build_image(T_i,tile_ids,epsilon,step_south_north):
    # Find dimensions
    H, W, tile_ids_real = calculate_grid_size(tile_ids,epsilon)

    # Get Grid 
    grid = get_grid(tile_ids_real,H,W,step_south_north)

    # Match Metadata with Grid 
    positions = match_tile_ids_with_grid(tile_ids_real,grid.reshape(-1))

    # Re-organize Tensor  
    new_T_i = resize_tensor(T_i, H,W, positions)
    new_T_i = new_T_i.reshape(new_T_i.size(0),new_T_i.size(1),H,W)
    return(new_T_i)

def get_station_data_and_permute_reshape(T,i):
    T_i = torch.squeeze(T[:,:,i,:,:])  #[Day, DL/UL, Tile-id, (hour,minute)] 
    # BIEN ETRE SUR DE CE PERMUTE RESHAPE §§§§
    T_i = T_i.permute(1,2,0,3)  #[DL/UL, Tile-id,Day,(hour,minute)] 
    T_i = T_i.reshape(T_i.size(0),T_i.size(1),-1) #[DL/UL, Tile-id, Day*(hour,minute)]
    #  ................................
    T_i = T_i.permute(2,0,1) #[Day*(hour,minute), DL/UL, Tile-id]   <-> [N,C,H*W]
    return(T_i)
    
    
def calculate_grid_size(tile_ids,epsilon):
    ''' Calculate Image dimension. Simple function cause Cell-size = 100m*100m '''
    # Remove tile ids '-1' 
    tile_ids_real =tile_ids[tile_ids> 0]
    
    H = W = 2*(epsilon//100 + 1)   # +1 to be sure. 2* cause epsilon is the radius 
    
    # === ... 
    return(H,W,tile_ids_real)

def get_grid(tile_ids_real,H,W,step_south_north = 287):
    ''' Lower-left corner (South - West) is supposed to be the minimal tile-id '''
    init_mini = tile_ids_real.min()
    mini = init_mini
    find = False

    # find 'mini', the lower_left_corner: 
    while not find:
    
        init_grid = np.array([[(mini + w + step_south_north*h) for w in range(W)]for h in range(H)])
        condition = set(tile_ids_real).issubset(set(init_grid.reshape(-1)))

        if condition:
            find = True
        else:
            mini = mini-1

        if mini < init_mini- 287:
            raise ValueError("Init Grid can't match Tile-Ids. There is an error within code.")

    return(init_grid)

def match_tile_ids_with_grid(v1,v2):    
    positions = [np.where(v2 == x)[0][0] for x in v1]
    return(positions)


def resize_tensor(T_i, H,W, positions):
    N,C,L = T_i.shape
    new_L = H*W
    new_T_i = torch.zeros((N,C,new_L))
    
    for idx,pos in enumerate(positions):
        new_T_i[:,:,pos] = T_i[:,:,idx]
        
    return(new_T_i)


def tackle_all_days(result,metadata,netmob_data_folder_path,app,maxi_nb_tile,folder_days,assert_transfer_mode= None ):
    '''
    For a specific app, but for each day, read Two CSV: UL/DL 
    
    outputs:
    --------
    4-th order Tensor [nb_days,N,nb_tiles,24H]
    
    N : number of spatial units (subway stations)
    nb_tiles:  number of cellules 100x100m
    24H: 96 time steps through each days
    '''
    # for each days  
    Tensors_days = []
    for day in folder_days:
        Tensors_days,metadata = tackl_one_day(result,metadata,netmob_data_folder_path,app,day,Tensors_days,maxi_nb_tile,assert_transfer_mode)
    Tensors_days = torch.stack(Tensors_days,dim=0)
    return(Tensors_days,metadata)

def tackl_one_day(result,metadata,netmob_data_folder_path,app,day,Tensors_days,maxi_nb_tile, assert_transfer_mode= None ):
    '''
    For a specific day and a specific app, read Two CSV: UL/DL 
    
    outputs:
    --------
    List of 2 elements (UL/DL) of a 3-th order Tensor [N,nb_tiles,24H]
    
    N : number of spatial units (subway stations)
    nb_tiles:  number of cellules 100x100m
    24H: 96 time steps through each days
    '''
    txt_paths = sorted(glob.glob(os.path.join(f'{netmob_data_folder_path}/{app}/{day}', "*.txt")))
    # For each transfert mode:
    Tensors_transfer,transfer_modes = [],[]
    for path in txt_paths:
        transfer_mode,columns = get_information_from_path(path)
        transfer_modes.append(transfer_mode)

        if (assert_transfer_mode is None) or (assert_transfer_mode == transfer_mode):
            Tensors,metadata = read_csv(path,result,metadata,maxi_nb_tile,columns)
            Tensors_transfer.append(Tensors)
        
    Tensors_transfer = torch.stack(Tensors_transfer,dim=0) 


    # Update metadata
    for station in result['COD_TRG']:
        metadata[station]['mode_transfer'] = transfer_modes
    Tensors_days.append(Tensors_transfer)
    return(Tensors_days,metadata)

def get_information_from_path(path):
    transfer_mode = path.split('.')[-2].split('_')[-1]
    day = path.split('_')[-2]
    day_str = str(day)
    day_str = datetime.strptime(day_str, '%Y%m%d')
    times = [day_str + timedelta(minutes=15*i) for i in range(96)]
    times_str = [t.strftime('%H:%M') for t in times]
    columns = ['tile_id'] + times_str
    return(transfer_mode,columns)


def read_csv(path,result,metadata,maxi_nb_tile,columns):
    '''
    Read a single CSV
    
    outputs:
    --------
    3-th order Tensor [N,nb_tiles,24H]
    
    N : number of spatial units (subway stations)
    nb_tiles:  number of cellules 100x100m
    24H: 96 time steps through each days
    '''

    Tensors = []
    df = pd.read_csv(path, sep = ' ', names = columns).set_index(['tile_id'])
    # Loop through each station. Get associated usefull Tile-id
    for station_ind in range(len(result)):
        station = result['COD_TRG'][station_ind]
        ids = result.tile_id[station_ind]
        #T =  torch.Tensor(df.loc[ids].values, dtype = torch.int32) 
        T =  torch.tensor(df.loc[ids].values, dtype = torch.int32)
        padding = maxi_nb_tile-T.size(0)

        if padding > 0:
            T = F.pad(T,(0,0,0,padding),'constant',0)
            ids = ids+[-1]*padding

        Tensors.append(T)
        metadata[station]['tile_id'] = ids
    Tensors = torch.stack(Tensors,dim = 0)

    return(Tensors,metadata)

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
    
    restrained_Lyon_gdf = gpd.sjoin(Netmob_gdf,working_zones,how = 'inner') #how = 'left')
    return(restrained_Lyon_gdf,working_zones)


def load_netmob_restrained_to_lyon(folder_path,data_folder,
                    geojson_path = 'Lyon.geojson',
                    zones_path = 'lyon_iris_shapefile/lyon.shp',
                    ):

    Netmob_gdf,n_rows,n_cols = load_netmob_json(folder_path, geojson_path)
    Netmob_gdf['centroid_lonlat'] = Netmob_gdf.geometry.apply(lambda x : x.centroid)
    Netmob_gdf = Netmob_gdf.set_geometry('centroid_lonlat')
    Netmob_gdf_joined,working_zones = restrain_netmob_to_Lyon(Netmob_gdf,data_folder,zones_path) #Associate an square to an IRIS when the centroid is inside it
    
    return(Netmob_gdf_joined,working_zones)

def load_netmob_gdf(folder_path = '../../Data/NetMob/',data_folder = '../../Data/lyon_iris_shapefile/', geojson_path = 'Lyon.geojson',zones_path = 'lyon.shp'):
    Netmob_gdf_joined,working_zones = load_netmob_restrained_to_lyon(folder_path,data_folder,geojson_path,zones_path)
    Netmob_gdf_joined = Netmob_gdf_joined[['tile_id','INSEE_COM','NOM_COM','NOM_IRIS','geometry']] # 
    Netmob_gdf_joined = gpd.GeoDataFrame(Netmob_gdf_joined,crs='EPSG:4326') #Don't know why, but the geodataframe seem corrupted as we can't convert it into GeoJson, that's why we need to use "gpd.GeoDataFrame()"

    return(Netmob_gdf_joined,working_zones)

if __name__ == '__main__':
    
    import pickle
    import torch

    # Init: 
    if torch.cuda.is_available():
        data_folder_path = '../../../../data/'
        save_folder = f"{data_folder_path}NetMob_tensor/"
        netmob_data_folder_path = f"{data_folder_path}NetMob/"
        PATH_iris = f'{data_folder_path}lyon_iris_shapefile/'
    else:
        data_folder_path = '../../../data/'
        save_folder = f"{data_folder_path}NetMob_tensor/"
        netmob_data_folder_path = f"{data_folder_path}NetMob/"
        PATH_iris = '../Data/lyon_iris_shapefile/'


    # Load Ref Subway: 
    ref_subway = load_subway_shp(folder_path = data_folder_path)

    # Parameters: size of netmob image 
    step_south_north = 287  # Incremente by 287-ids when passing from south to north. 
    epsilon=1000  #epsilon : radius, in meter (1000m) 
    # W,H = 2*(epsilon//100 + 1), 2*(epsilon//100 + 1)

    '''
    Define the NetMob Geodatarame associated to Lyon City.
    Build 'result' which keep track on tile-ids associated to each subway stations
    '''
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Load subway gdf adn NetMob gdf
    Netmob_gdf,working_zones = load_netmob_gdf(folder_path = netmob_data_folder_path,
                                data_folder = PATH_iris, 
                                geojson_path = 'Lyon.geojson',
                                zones_path = 'lyon.shp')
    Netmob_gdf_dropped = Netmob_gdf.drop_duplicates(subset = ['tile_id'])  # Some Doubles are exis

    # Get Cell-Id within epsilon : 
    result,joined = find_ids_within_epsilon(Netmob_gdf_dropped,ref_subway,epsilon=epsilon) 
    maxi_nb_tile =  result.apply(lambda row: len(row.tile_id),axis=1).max()
    print(f"Maximum number of NetMob Cell associated to a subway station: {maxi_nb_tile}")


    '''
    Load NetMob Data From raw Data
    Build 4-th order Tensor :  [nb_days,N,nb_tiles,24H]

        nb_days: number of available days (from 16 march to 31 May 2019)
        N : number of spatial units (subway stations)
        nb_tiles:  number of cellules 100x100m
        24H: 96 time steps through each days

    ''' 
    apps = [app for app in listdir(netmob_data_folder_path) if ((app != 'Lyon.geojson') and (not app.startswith('.'))) ]   # Avoid hidden folder and Lyon.geojson
    Tensors = []
    # For each app
    for app in apps: 
        print('App: ',app)
        metadata = {result['COD_TRG'][station_ind] : {} for station_ind in range(len(result))}
        folder_days = [day for day in listdir(f'{netmob_data_folder_path}/{app}') if (not day.startswith('.')) ]
        Tensors_days,metadata = tackle_all_days(result,metadata,netmob_data_folder_path,app,maxi_nb_tile,folder_days)
        torch.save(Tensors_days,f"{save_folder}{app}.pt")
        pickle.dump(metadata,open(f"{save_folder}{app}_metadata.pkl",'wb'))

    #Load Tensor example:
    app = 'Apple_Video'

    Apple_Video_meta = pickle.load(open(f"{save_folder}{app}_metadata.pkl","rb"))
    Apple_Video = torch.load(f"{save_folder}{app}.pt")  #[day, transfer_mode, Station, Tile_id, (hour,minutes)]
    print(f"{app} Tensor: {Apple_Video.size()}")
    # ....


    ''' 
    Build and Save NetMob Tensor Image associated to a station 'i': 
    NetMob Tensor Image: 5-th order Tensor [T]
    '''
    for i in range(len(ref_subway)):
        List_channel_of_station_i = []
        for app in apps: 

            metadata = pickle.load(open(f"{save_folder}{app}_metadata.pkl","rb"))
            T = torch.load(f"{save_folder}{app}.pt")  #[day, transfer_mode, Station, Tile_id, (hour,minutes)]

            stations = list(metadata.keys())
            station = stations[i]
            tile_ids = np.array(metadata[station]['tile_id'])

            T_i = get_station_data_and_permute_reshape(T,i)
            resized_T_i = build_image(T_i,tile_ids,epsilon,step_south_north)
            List_channel_of_station_i.append(resized_T_i)


        name_save = f"station_{station}"
        print(name_save)
        Station_i_with_all_channel = torch.cat(List_channel_of_station_i, dim=1)
        torch.save(Station_i_with_all_channel,f"{save_folder}{name_save}.pt")

    # Load the saved image : 
    station = 'AMP'
    T_amp = torch.load(f"{save_folder}station_{station}.pt")  #[day, transfer_mode, Station, Tile_id, (hour,minutes)]
    T_amp.size()
    # ...
