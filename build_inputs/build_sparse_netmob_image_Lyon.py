import pickle
from build_netmob_data import load_subway_shp,load_netmob_gdf,get_information_from_path
import torch
import pandas as pd
import os 
import sys
import glob
import numpy as np
# Obtenir le chemin du dossier parent
current_path = notebook_dir = os.getcwd()
# current_path = os.path.dirname()
working_dir = os.path.abspath(os.path.join(current_path, '..'))

# Ajouter le dossier parent au chemin de recherche des modules
if working_dir not in sys.path:
    sys.path.insert(0, working_dir)


def tackle_one_days_entire_map(txt_path,Lyon_ids,P,netmob_data_folder_path,app,day,transfer_mode,columns):
    #Read CSV
    txt_path = glob.glob(os.path.join(f'{netmob_data_folder_path}/{app}/{day}',f"*_{transfer_mode}.txt"))[0]
    transfer_mode,columns = get_information_from_path(txt_path)
    df = pd.read_csv(txt_path, sep = ' ', names = columns).set_index(['tile_id'])
    df = df.loc[Lyon_ids]
    
    k0,kn,N =get_windows_caracteristics(df)

    # Get df within rectangular window (filled with 0 when empty)
    new_df = df.reindex(np.arange(kn,k0,-1),fill_value=0)

    # Build Tensor
    T = torch.tensor(new_df.values, dtype = torch.int32)
    T = T.permute(1,0)
    T = T.reshape(T.size(0),N,P)
    return(T)
        
def find_N(km,kM,P = 287):
    ''' 
    args 
    ---------
    km,kM : tile_id min and tile_id max
    P: number of tiles per row
    
    outputs : 
    N : number of tiles per colum
    '''
    N = (kM-km)//P + 1  # N = N_ if  rest == 0
    
    # Particular case when kM < km + N_*P
    if (N != (kM-km)/P) and (kM < km + N*P):
        N = N + 1
            
    return N

def get_limits_tile_id(km,N,P):
    first_tile_id = (km//P)*P
    last_tile_id = first_tile_id+ N*P
    
    return first_tile_id,last_tile_id

def get_windows_caracteristics(df,P = 287):
    km,kM = df.index.min(),df.index.max()
    
    # Find limits (the window)
    N = find_N(km,kM,P)
    k0,kn = get_limits_tile_id(km,N,P)
    return(k0,kn,N)
        

def plotting_verification(T00):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.colors as mcolors

    # Créer un masque pour les cellules avec des valeurs non-nulles
    mask_zero = (T00 == 0)

    # Créer une figure et un axe
    plt.figure(figsize=(10, 10))

    # Afficher les cellules non nulles avec la colormap viridis
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=T00[T00 > 0].min().item(), vmax=T00.max().item())

    # Afficher les cellules avec des valeurs non-nulles
    plt.imshow(np.ma.masked_where(T00 == 0, T00), cmap=cmap, norm=norm, origin='upper', interpolation='none')

    # Afficher les cellules contenant 0 en gris
    plt.imshow(np.ma.masked_where(T00 != 0, T00), cmap=plt.cm.Greys, origin='upper', interpolation='none')

    # Ajouter une barre de couleurs pour les cellules non nulles
    plt.colorbar(label='Amplitude des valeurs')

    # Ajouter un titre et des labels
    plt.title('Visualisation du Tensor avec les cellules à 0 en gris')
    plt.xlabel('X (cellules)')
    plt.ylabel('Y (cellules)')

    # Afficher le graphique
    plt.show()


if __name__ == '__main__':
    '''Looking to build the Tensor image associated to df. 
    The image dimension is [N,P]. 
    Between each row there is 287 Tile.
    '''


    # Init: ========================================
    data_folder_path = '../../../../data/'
    save_folder = f"{data_folder_path}NetMob_tensor/"
    netmob_data_folder_path = f"{data_folder_path}NetMob/"
    PATH_iris = f'{data_folder_path}lyon_iris_shapefile/'
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
    #Netmob_gdf_dropped.explore()

    #       ========================================

    
    transfer_mode = 'DL'
    Lyon_ids = Netmob_gdf_dropped.tile_id
    apps = [app for app in os.listdir(netmob_data_folder_path) if ((app != 'Lyon.geojson') and (not app.startswith('.'))) ]   # Avoid hidden folder and Lyon.geojson


    # For each app
    T_apps = []
    import time
    
    t0 = time.time()
    for app in apps: 
        T_days = []
        folder_days = [day for day in os.listdir(f'{netmob_data_folder_path}/{app}') if (not day.startswith('.'))] 
        for day in folder_days:
            T = tackle_one_days_entire_map(txt_path,Lyon_ids,P,netmob_data_folder_path,app,day,transfer_mode,columns)
            T_days.append(T)
        T_days = torch.cat(T_days)
        T_apps.append(T_days)
    T_apps = torch.stack(T_apps,dim=0)
    print('T_apps: ', T_apps.size(),'time: ', time.time()-t0)
    # ...


    name_save = 'NetMob_DL_video_Lyon'
    save_path = '../../../../data'

    torch.save(T_apps,f"{save_path}/{name_save}.pt")
    pickle.dump(apps,open(f"{save_path}/{name_save}_APP.pkl",'wb'))

    # If Plotting verification :
    T00 = T_days[7*4]  #7:00 am
    plotting_verification(T00)