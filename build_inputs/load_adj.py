import pandas as pd 
import os
import sys
current_file_path = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(current_file_path,'..'))
if ROOT not in sys.path:
    sys.path.insert(0,ROOT)
from constants.paths import ABS_PATH_PACKAGE,FOLDER_PATH

def load_adj(dataset,folder = 'adj',adj_type = 'adj',threshold = None):
    '''
    args : 
    --------
    adj_type = choice ['dist','corr']
    If i and j distant, then value -> 0, else, value -> 1. 
        - 'corr' is based on Pearson Correlation Coefficient (PCC). 
            Then, gso = abs(Correlation-Matrix([v1,...,vn]))
        - 'dist' is based on gaussian kernel 
           Then, gso = exp(-dist(u,v)^2 / sigma^2)
    '''
    # Correlation matrix Must be different according to the considered train set
    if adj_type == 'corr':
        gso =  abs(pd.DataFrame(dataset.train_input).corr())
        assert threshold is not None, f"You defined a distance-based (adj_type: {adj_type}) matrix but you did not define any threshold distance"
        gso[gso < threshold] = 0 
    # Otherwise, we can load the precomputed weighted adj matrix:
    else:
        gso = pd.read_csv(f'{ABS_PATH_PACKAGE}/{FOLDER_PATH}/{dataset.target_data}/{folder}/{adj_type}.csv',index_col = 0)
        gso = gso.iloc[dataset.indices_spatial_unit]
        if (adj_type == 'dist') :
            assert threshold is not None, f"You defined a distance-based (adj_type: {adj_type}) matrix but you did not define any threshold distance"
            gso[gso < threshold] = 0 
    
    n_vertex = len(gso)
    return(gso,n_vertex)