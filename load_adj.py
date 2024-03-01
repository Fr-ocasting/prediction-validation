import pandas as pd 

def load_adj(folder = 'subway_adj',adj_type = 'adj'):
    '''
    args : 
    --------
    adj_type = choice ['dist','corr']
        - 'corr' is based on Pearson Correlation Coefficient (PCC)
        - 'dist' is based on gaussian kernel exp(-dist(u,v)^2 / sigma^2)
    '''
    gso = pd.read_csv(f'data/{folder}/{adj_type}.csv',index_col = 0)
    n_vertex = len(gso)
    return(gso,n_vertex)