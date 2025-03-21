import sys 
import os 
import pandas as pd
import torch 

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
from datetime import datetime 
from dataset import DataSet
from dataset import PersonnalInput
from sklearn.cluster import AgglomerativeClustering
import numpy as np 
import pickle
from utils.utilities import filter_args
from build_inputs.load_netmob_data import find_positions,replace_heure_d_ete
''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - >>>> No Need to set n_vertex as it's a contextual data 
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''


FILE_NAME = 'netmob_image_per_station'
START = '03/16/2019'
END = '06/01/2019'
FREQ = '15min'

list_of_invalid_period = []
list_of_invalid_period.append([datetime(2019,5,16,0,0),datetime(2019,5,16,18,15)])  # 16 mai 00:00 - 18:15
list_of_invalid_period.append([datetime(2019,5,11,23,15),datetime(2019,5,12,0,0)])  # 11 mai 23:15 - 11 mai 23:59: down META (fb, whatsapp)
list_of_invalid_period.append([datetime(2019,5,23,0,0),datetime(2019,5,25,6,0)])  # Anoamlies for every single apps  23-25 May



## C = 1
## n_vertex = 

def load_data(dataset,ROOT,FOLDER_PATH,invalid_dates,intersect_coverage_period,args,normalize= True): # args,ROOT,FOLDER_PATH,coverage_period = None
    '''
    args:
    ------
    restricted : if True, then the used map is at least 4 times smaller and not sparse anymore
    normalize : supposed to be = True

    outputs:
    --------
    4-th order tensor [len(apps),len(osmid),len(transfer_modes),len(time-serie)]
    '''

    id_stations = dataset.spatial_unit
    NetMob_ds = []
    nb_pois_by_station = []
    for id_station in id_stations:
        # data_app.shape :[len(apps),len(osmid_associated),len(transfer_modes),T]
        netmob_T = torch.Tensor(load_data_npy(id_station,ROOT,FOLDER_PATH,args))
        netmob_T = netmob_T.permute(3,0,1,2)

        # Extract only usefull data, and replace "heure d'été"
        netmob_T = replace_heure_d_ete(netmob_T,start = 572, end = 576)

        # Temporal Aggregation if needed: 
        if args.freq != FREQ :
            assert int(args.freq.replace('min',''))> int(FREQ.replace('min','')), f'Trying to apply a a {args.freq} temporal aggregation while the minimal possible one is {FREQ}'
            netmob_T = netmob_T.view(-1, int(args.freq.replace('min','')) // int(FREQ.replace('min','')), *netmob_T.shape[1:]).sum(dim=1)
        
        # Extract only usefull data : [T,R] -> [T',R]
        coverage_local = pd.date_range(start=START, end=END, freq=args.freq)[:-1]
        indices_dates = [k for k,date in enumerate(coverage_local) if date in intersect_coverage_period]
        netmob_T = netmob_T[indices_dates]

        # Reduce dimensionality : 
        netmob_T = netmob_T.reshape(netmob_T.size(0),-1)

        # REMOVE THE DIMENSION REDUCTION CAUSE CORRELATION BASED ON THE ENTIRE DATASET. SHOULD BE BASED ONLY ON TRAIN  
        if False:
            netmob_T = reduce_dim_by_clustering(pd.DataFrame(netmob_T),epsilon = args.epsilon_clustering)
            netmob_T = torch.Tensor(netmob_T.values)
         # REMOVE THE DIMENSION REDUCTION CAUSE CORRELATION BASED ON THE ENTIRE DATASET. SHOULD BE BASED ONLY ON TRAIN    
         #  
         
        # netmob_T.shape : [T,len(apps),len(osmid_associated),len(transfer_modes)]
        #netmob_T = netmob_T.permute(3,0,1,2)
        # netmob_T.shape : [T,len(apps)*len(osmid_associated)*len(transfer_modes)] = [T,R]
        #netmob_T = netmob_T.reshape(netmob_T.size(0),-1)
        


        # dimension on which we want to normalize: 
        dims = [0]# [0]  -> We are normalizing each time-serie independantly 
        NetMob_POI = load_input_and_preprocess(dims = dims,normalize=normalize,invalid_dates=invalid_dates,args=args,netmob_T=netmob_T,dataset=dataset)
        NetMob_POI.station_name = id_station
        NetMob_POI.periods = None # dataset.periods
        NetMob_POI.spatial_unit = list(np.arange(netmob_T.size(1)))
        NetMob_ds.append(NetMob_POI)


        nb_pois_by_station.append(netmob_T.size(1))

    print('Custom POIs associated by stations: ',[f"{id_station}: {nb_pois_by_station[k]}" for k,id_station in enumerate(id_stations)])
    return(NetMob_ds)


def load_data_npy(id_station,ROOT,FOLDER_PATH,args):
    if hasattr(args,'NetMob_only_epsilon') and getattr(args,'NetMob_only_epsilon'):
        save_folder = f"{ROOT}/{FOLDER_PATH}/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}"
    else:
        save_folder = f"{ROOT}/{FOLDER_PATH}/POIs/netmob_POI_Lyon{args.NetMob_expanded}/Inputs/{id_station}"
    data_app = np.load(open(f"{save_folder}/data.npy","rb"))
    metadata = pickle.load(open(f"{save_folder}/metadata.pkl","rb"))

    pos_selected_apps = [k for k,app in enumerate(metadata['apps']) if app in args.NetMob_selected_apps]
    pos_selected_modes = [k for k,mode in enumerate(metadata['transfer_modes']) if mode in args.NetMob_transfer_mode]
    pos_selected_tags = [k for k,tag in enumerate(metadata['tags']) if tag in args.NetMob_selected_tags]

    # Extract sub-dataset thanks to np.ix_ (kind of n-dimensional meshgrid)
    n1 = np.array(pos_selected_apps)
    n2 = np.array(pos_selected_tags)
    n3 = np.array(pos_selected_modes)
    n4 = np.arange(data_app.shape[-1])
    idxs = [n1,n2,n3,n4]
    sub_indices = np.ix_(*idxs)
    data_app = data_app[sub_indices]  # data_app[pos_selected_apps,pos_selected_tags,pos_selected_modes,:]

    return(data_app)


def load_input_and_preprocess(dims,normalize,invalid_dates,args,netmob_T,dataset):
    
    args_DataSet = filter_args(DataSet, args)

    NetMob_ds = PersonnalInput(invalid_dates,args, tensor = netmob_T, dates = dataset.df_dates,
                           time_step_per_hour = dataset.time_step_per_hour,
                           minmaxnorm = True,
                           dims =dims,
                           **args_DataSet)
    
    NetMob_ds.preprocess(args.train_prop,args.valid_prop,args.test_prop,args.train_valid_test_split_method,normalize)

    return NetMob_ds


def agg_clustering(multi_ts,epsilon):
    clustering = AgglomerativeClustering(
        n_clusters=None,  # 
        metric='precomputed',  # We are already using a disance matrix, so no need to compute it 
        linkage='complete',  # 'complete' to assert the 'distance max' is kept within every cluster 
        distance_threshold=epsilon  # 'distance max' threshold
    )
    # define distance matrix
    df_distance = 1-abs(multi_ts.corr())

    # Get labels
    labels = clustering.fit_predict(df_distance.values)

    return(labels)

def reduce_dim_by_clustering(multi_ts,epsilon,agg_function='median'):
    '''
    args:
    -----
    multi_ts: DataFrame multi time-serie. 
    epsilon: maximum accepted distance correlation as diameter within cluster (agglomerative clustering)
    '''
    labels = agg_clustering(multi_ts,epsilon = epsilon)
    unique_labels = list(set(labels))

    #df_reduced = pd.DataFrame()
    #for label in unique_labels:
    #    columns= [k for k,lab in enumerate(labels) if lab == label]
    #    ts_rpz_label = multi_ts[columns].max(axis=1)
    #    df_reduced[label] = ts_rpz_label

    if agg_function == 'median':
        df_reduced = pd.DataFrame({label: multi_ts[[k for k,lab in enumerate(labels) if lab == label]].median(axis=1) for label in unique_labels})
    elif agg_function == 'mean':
        df_reduced = pd.DataFrame({label: multi_ts[[k for k,lab in enumerate(labels) if lab == label]].mean(axis=1) for label in unique_labels})
    elif agg_function == 'max':
        df_reduced = pd.DataFrame({label: multi_ts[[k for k,lab in enumerate(labels) if lab == label]].max(axis=1) for label in unique_labels})
    else:
        raise NotImplementedError

    return(df_reduced)

