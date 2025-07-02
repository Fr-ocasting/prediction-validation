import sys 
import os 
import pandas as pd
import torch 

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
from datetime import datetime 
from sklearn.cluster import AgglomerativeClustering
import numpy as np 
from build_inputs.load_contextual_data import replace_heure_d_ete
from build_inputs.load_preprocessed_dataset import load_input_and_preprocess

''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - >>>> No Need to set num_nodes as it's a contextual data 
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''

NAME = 'netmob_POIs'
#FILE_NAME = 'netmob_image_per_station'
START = '03/16/2019'
END = '06/01/2019'
USELESS_DATES = {'hour':[], #[1,2,3,4,5,6],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }
FREQ = '15min'

list_of_invalid_period = []
list_of_invalid_period.append([datetime(2019,5,16,0,0),datetime(2019,5,16,18,15)])  # 16 mai 00:00 - 18:15
list_of_invalid_period.append([datetime(2019,5,11,23,15),datetime(2019,5,12,0,0)])  # 11 mai 23:15 - 11 mai 23:59: down META (fb, whatsapp)
list_of_invalid_period.append([datetime(2019,5,23,0,0),datetime(2019,5,25,6,0)])  # Anoamlies for every single apps  23-25 May



## C = 1
## num_nodes = 
def load_data(FOLDER_PATH,invalid_dates,coverage_period,args,minmaxnorm,standardize,normalize= True,tensor_limits_keeper = None,): # args,FOLDER_PATH,coverage_period = None
    '''
    args:
    ------
    restricted : if True, then the used map is at least 4 times smaller and not sparse anymore
    normalize : supposed to be = True

    outputs:
    --------
    PersonalInput object. Containing a 2-th order tensor [T,R]
    '''

    # data_app.shape :[nb-osmid*(apps*transfer_mode*tags), T]
    netmob_T = load_data_npy(FOLDER_PATH,args)

    # [T,nb-osmid*(apps*transfer_mode*tags)]
    netmob_T = netmob_T.permute(1,0)

    # Extract only usefull data, and replace "heure d'été"
    netmob_T = replace_heure_d_ete(netmob_T,start = 572, end = 576)

    # Temporal Aggregation if needed: 
    if False: 
        if args.freq != FREQ :
            assert int(args.freq.replace('min',''))> int(FREQ.replace('min','')), f'Trying to apply a a {args.freq} temporal aggregation while the minimal possible one is {FREQ}'
            netmob_T = netmob_T.view(-1, int(args.freq.replace('min','')) // int(FREQ.replace('min','')), *netmob_T.shape[1:]).sum(dim=1)
        
        # Extract only usefull data : [T,R] -> [T',R]
        coverage_local = pd.date_range(start=START, end=END, freq=args.freq)[:-1]
    coverage_local = pd.date_range(start=START, end=END, freq=FREQ)[:-1] 

    # Allow to deal with 2 source with different temporal aggregation
    if args.freq != FREQ:
        indices_dates = [k for k,date in enumerate(coverage_local) if date >= min(coverage_period) and date <= max(coverage_period)]
        coverage_period = pd.date_range(start=min(coverage_period), end=max(coverage_period), freq=FREQ)
        STEP_AHEAD = args.step_ahead * int(args.freq.replace('min','')) // int(FREQ.replace('min','')) 
        HORIZON_STEP = args.horizon_step * int(args.freq.replace('min','')) // int(FREQ.replace('min',''))
    else:
        indices_dates = [k for k,date in enumerate(coverage_local) if date in coverage_period]
        STEP_AHEAD = args.step_ahead
        HORIZON_STEP = args.horizon_step

    netmob_T = netmob_T[indices_dates]
    """
    local_df_dates = pd.DataFrame(coverage_local[indices_dates])
    local_df_dates.columns = ['date']
    """

    # Reduce dimensionality : [T',R] -> [T',R']
    # REMOVE THE DIMENSION REDUCTION CAUSE CORRELATION BASED ON THE ENTIRE DATASET. SHOULD BE BASED ONLY ON TRAIN  
    if 'epsilon_clustering' in args.contextual_kwargs['netmob_POIs'].keys() and args.contextual_kwargs['netmob_POIs']['epsilon_clustering'] is not None:
        print('    ATTENTION: Dimension reduction by clustering is applied on the entire dataset. This should be done only on the training set.')
        initial_size = netmob_T.size(-1)
        netmob_T = reduce_dim_by_clustering(netmob_T,epsilon = args.contextual_kwargs['netmob_POIs']['epsilon_clustering'])
        print(f"    Netmob_T.size(): {netmob_T.size()}. Dimensionality reduced by {'{:.1%}'.format(netmob_T.size(-1)/initial_size)}")
    # REMOVE THE DIMENSION REDUCTION CAUSE CORRELATION BASED ON THE ENTIRE DATASET. SHOULD BE BASED ONLY ON TRAIN  
    
    # dimension on which we want to normalize: 
    dims = [0]# [0]  -> We are normalizing each time-serie independantly 
    NetMob_POI = load_input_and_preprocess(dims = dims,normalize=normalize,invalid_dates=invalid_dates,
                                           args=args,data_T=netmob_T,coverage_period = coverage_period,
                                           freq = FREQ,step_ahead = STEP_AHEAD, horizon_step = HORIZON_STEP,
                                           name=NAME,
                                           minmaxnorm=minmaxnorm,
                                           standardize=standardize,
                                           tensor_limits_keeper=tensor_limits_keeper) 
    NetMob_POI.periods = None # dataset.periods
    NetMob_POI.spatial_unit = list(np.arange(netmob_T.size(1)))

    
    return(NetMob_POI)


def load_data_npy(FOLDER_PATH,args):
    save_folder = f"{FOLDER_PATH}/POIs/netmob_POI_Lyon{args.contextual_kwargs['netmob_POIs']['NetMob_expanded']}/Inputs/agg_TS"
    list_of_data = []

    if len(args.contextual_kwargs['netmob_POIs']['NetMob_selected_apps']) == 0:
        raise ImportError("No NetMob apps selected. Please check the configuration of netmob_POIs in the arguments.")
    if len(args.contextual_kwargs['netmob_POIs']['NetMob_selected_tags']) == 0:
        raise ImportError("No NetMob tags selected. Please check the configuration of netmob_POIs in the arguments.")
    if len(args.contextual_kwargs['netmob_POIs']['NetMob_transfer_mode']) == 0:
        raise ImportError("No NetMob transfer mode selected. Please check the configuration of netmob_POIs in the arguments.")
    
    # Load data for each app, transfer mode and tag
    for app in args.contextual_kwargs['netmob_POIs']['NetMob_selected_apps']:
        for mode in args.contextual_kwargs['netmob_POIs']['NetMob_transfer_mode']:
            for tag in args.contextual_kwargs['netmob_POIs']['NetMob_selected_tags']:
                folder_path_to_save_agg_data = f"{save_folder}/{tag}/{app}/{mode}"
                try: 
                    list_of_data.append(torch.Tensor(np.load(open(f"{folder_path_to_save_agg_data}/data.npy","rb"))))  # [nb-osmid, T]
                except:
                    raise ImportError(f"{folder_path_to_save_agg_data}/data.npy does not exist. NetMob app {app} Might not have been recorded.")
    netmob_T = torch.cat(list_of_data)    # [nb-osmid*(apps*transfer_mode*tags), T]

    #print(args.NetMob_selected_apps,args.NetMob_transfer_mode,args.NetMob_selected_tags)
    #print(netmob_T.size())
    return netmob_T


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

