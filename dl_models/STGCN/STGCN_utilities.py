import scipy.sparse as sp
import numpy as np 
from scipy.sparse.linalg import norm
from scipy.sparse.linalg import svds

def calc_gso(dir_adj, gso_type):
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # Symmetrizing an adjacency matrix
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
    #adj = 0.5 * (dir_adj + dir_adj.transpose())

    # Dans certains cas, ajoute l'identité (A = A+I)
    if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + id

    #  Dans certains cas (y compris dans les précédent A+I), on produit Ã = sqrt(D)*A*sqrt(D)
    if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
        or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
        row_sum = adj.sum(axis=1).A1
        row_sum_float = row_sum.astype(float)
        row_sum_inv_sqrt = np.power(row_sum_float, -0.5)# Liste de N éléments 
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_{sym} = D^{-0.5} * A * D^{-0.5}
        #sym_norm_adj = sp.csr_matrix.dot(sp.csc_matrix.dot(deg_inv_sqrt,adj), deg_inv_sqrt)
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            sym_norm_lap = id - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    # Dans certain cas (y compris dans les précédent A+I)  on produit Ã = D^-1 A
    elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
        row_sum = np.sum(adj, axis=1).A1    # Liste de N éléments   . .A1 permet de passer de (N,1) à (N,) 
        row_sum_float = row_sum.astype(float)
        row_sum_inv = np.power(row_sum_float, -1)# Liste de N éléments 
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        #deg_inv = np.diag(row_sum_inv)    # Matrice diag [N,N]
        deg_inv = sp.diags(row_sum_inv,format ='csc')
        # A_{rw} = D^{-1} * A
        rw_norm_adj = deg_inv.dot(adj)
        #rw_norm_adj = sp.csc_matrix.dot(deg_inv,adj)  

        if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            rw_norm_lap = id - rw_norm_adj
            gso = rw_norm_lap

        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')
    return gso

def calc_chebynet_gso(gso):
    if sp.issparse(gso) == False:
        gso = sp.csc_matrix(gso)
    elif gso.format != 'csc':
        gso = gso.tocsc()

    id = sp.identity(gso.shape[0], format='csc')
    # If you encounter a NotImplementedError, please update your scipy version to 1.10.1 or later.
    try : 
        eigval_max = norm(gso, 2)
    except : 
        _, s, _ = svds(gso, k=1, solver="lobpcg")      # If NotImplementedError, here the missing lines for norm(gso,ord=2)
        eigval_max =  s[0]

    # If the gso is symmetric or random walk normalized Laplacian,
    # then the maximum eigenvalue is smaller than or equals to 2.
    if eigval_max >= 2:
        gso = gso - id
    else:
        gso = 2 * gso / eigval_max - id

    return gso
