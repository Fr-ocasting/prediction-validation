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
    PersonalInput object. Containing a 2-th order tensor [T,R]
    '''

    # data_app.shape :[len(apps)*len(osmid_associated)*len(transfer_modes),T]
    netmob_T = load_data_npy(ROOT,FOLDER_PATH,args)

    # [T,len(apps)*len(osmid_associated)*len(transfer_modes)]
    netmob_T = netmob_T.permute(1,0)

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
    local_df_dates = pd.DataFrame(coverage_local[indices_dates])
    local_df_dates.columns = ['date']
    # Reduce dimensionality : [T',R] -> [T',R']

    
    # REMOVE THE DIMENSION REDUCTION CAUSE CORRELATION BASED ON THE ENTIRE DATASET. SHOULD BE BASED ONLY ON TRAIN  
    netmob_T = reduce_dim_by_clustering(netmob_T,epsilon = args.epsilon_clustering)
    # REMOVE THE DIMENSION REDUCTION CAUSE CORRELATION BASED ON THE ENTIRE DATASET. SHOULD BE BASED ONLY ON TRAIN  

    # dimension on which we want to normalize: 
    dims = [0]# [0]  -> We are normalizing each time-serie independantly 
    NetMob_POI = load_input_and_preprocess(dims = dims,normalize=normalize,invalid_dates=invalid_dates,args=args,netmob_T=netmob_T,dataset=dataset,df_dates = local_df_dates)
    NetMob_POI.periods = None # dataset.periods
    NetMob_POI.spatial_unit = list(np.arange(netmob_T.size(1)))
    
    return(NetMob_POI)


def load_data_npy(ROOT,FOLDER_PATH,args):
    save_folder = f"{ROOT}/{FOLDER_PATH}/POIs/netmob_POI_Lyon{args.NetMob_expanded}/Inputs/agg_TS"
    list_of_data = []
    for app in args.NetMob_selected_apps:
        for mode in args.NetMob_transfer_mode:
            for tag in args.NetMob_selected_tags:
                folder_path_to_save_agg_data = f"{save_folder}/{tag}/{app}/{mode}"
                list_of_data.append(torch.Tensor(np.load(open(f"{folder_path_to_save_agg_data}/data.npy","rb"))))  # [nb-osmid, T]
    netmob_T = torch.cat(list_of_data)    # [nb-osmid*(apps*transfer_mode*tags), T]
    return netmob_T


def load_input_and_preprocess(dims,normalize,invalid_dates,args,netmob_T,dataset,df_dates=None):
    if df_dates is None:
        df_dates = dataset.df_dates
    args_DataSet = filter_args(DataSet, args)
    #print('Netmb_T.size: ',netmob_T.size())
    #print('df_dates: ',dataset.df_dates)
    #print('Theoric df-dates length:',len(pd.date_range(start=START, end=END, freq=args.freq)[:-1]))
    #blabla
    NetMob_ds = PersonnalInput(invalid_dates,args, tensor = netmob_T, dates = df_dates,
                           time_step_per_hour = dataset.time_step_per_hour,
                           minmaxnorm = True,
                           dims =dims,
                           **args_DataSet)
    
    NetMob_ds.preprocess(args.train_prop,args.valid_prop,args.test_prop,args.train_valid_test_split_method,normalize)

    return NetMob_ds


def agg_clustering(multi_ts,epsilon):
    '''
    multi_ts: torch.Tensor of dimension [T,P]. 
    T: nb of time slots
    P: nb of features
    '''
    clustering = AgglomerativeClustering(
        n_clusters=None,  # 
        metric='precomputed',  # We are already using a disance matrix, so no need to compute it 
        linkage='complete',  # 'complete' to assert the 'distance max' is kept within every cluster 
        distance_threshold=epsilon  # 'distance max' threshold
    )
    # define distance matrix
    corr_matrix = torch.corrcoef(multi_ts.permute(1,0))
    dist_matrix = 1-abs(corr_matrix)

    # Get labels
    labels = clustering.fit_predict(dist_matrix.cpu().numpy())

    return(labels)

def reduce_dim_by_clustering(multi_ts,epsilon,agg_function = 'median'):
    '''
    args:
    -----
    multi_ts: torch.Tensor of dimension [T,P]. 
    epsilon: maximum accepted distance correlation as diameter within cluster (agglomerative clustering)
    '''
    labels = agg_clustering(multi_ts,epsilon = epsilon)

    unique_labels = sorted(set(labels))
    cluster_aggregates = []
    for label in unique_labels:
        col_indices = [idx for idx, lab in enumerate(labels) if lab == label]

        # [T, len(col_indices)]
        cluster_data = multi_ts[:, col_indices]

        if agg_function == 'median':
            agg_tensor = cluster_data.median(dim=1).values  
        elif agg_function == 'mean':
            agg_tensor = cluster_data.mean(dim=1)
        elif agg_function == 'max':
            agg_tensor = cluster_data.max(dim=1).values
        else:
            raise NotImplementedError(f"Not Supported Aggregation")
        
        cluster_aggregates.append(agg_tensor)
    tensor_reduced = torch.stack(cluster_aggregates, dim=1)

    return(tensor_reduced)

