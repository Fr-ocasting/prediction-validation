import pandas as pd 
import os
import sys
current_file_path = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(current_file_path,'..'))
if ROOT not in sys.path:
    sys.path.insert(0,ROOT)
from constants.paths import ABS_PATH_PACKAGE,FOLDER_PATH,DATA_TO_PREDICT

def load_adj(folder = 'subway_adj',adj_type = 'adj'):
    '''
    args : 
    --------
    adj_type = choice ['dist','corr']
        - 'corr' is based on Pearson Correlation Coefficient (PCC)
        - 'dist' is based on gaussian kernel exp(-dist(u,v)^2 / sigma^2)
    '''
    gso = pd.read_csv(f'{ABS_PATH_PACKAGE}/{FOLDER_PATH}/{DATA_TO_PREDICT}/{folder}/{adj_type}.csv',index_col = 0)
    n_vertex = len(gso)
    return(gso,n_vertex)