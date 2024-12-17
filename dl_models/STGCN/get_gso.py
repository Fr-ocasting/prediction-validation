
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

import torch 
import numpy as np 

from build_inputs.load_adj import load_adj
from dl_models.STGCN.STGCN_utilities import calc_chebynet_gso,calc_gso
from constants.paths import DATA_TO_PREDICT

def is_condition(args):
    condition1 = not(len(vars(args.args_vision))>0) and not(getattr(args.args_embedding,'concatenation_early'))
    condition2 = not(len(vars(args.args_embedding))>0) and not(getattr(args.args_vision,'concatenation_early'))
    condition3 = ((len(vars(args.args_embedding))>0) and (len(vars(args.args_vision))>0)) and (not(getattr(args.args_embedding,'concatenation_early')) and not(getattr(args.args_vision,'concatenation_early')))
    return(condition1 or condition2 or condition3)


def get_output_kernel_size(args):
     # Set Ko : Last Temporal Channel dimension before passing through output module :
    if args.enable_padding: 
        Ko = args.L  if args.L > 0 else 1
    else :
        Ko = args.L - (args.Kt - 1) * 2 * args.stblock_num   

    # Tackle the case where prediction is based only from contextual data:
    if  not (DATA_TO_PREDICT in args.dataset_names):
        Ko = 0
        if is_condition(args):
            Ko = 1
        ''' si pas de vision, mais calendar: 
        if not(len(vars(args.args_vision))>0):
            # Si on concatène seulement en sortie et pas en entrée : 
            if not(getattr(args.args_embedding,'concatenation_early')):
                Ko = 1
        # si pas de calendar, mais vision:    
        if not(len(vars(args.args_embedding))>0):
            # Si on concatène seulement en sortie et pas en entrée : 
            if not(getattr(args.args_vision,'concatenation_early')):
                Ko = 1
        # Si calendar et vision,
        if (len(vars(args.args_embedding))>0) and (len(vars(args.args_vision))>0):
            #et qu'aucun des deux de concatène en entrée:     
            if not(getattr(args.args_embedding,'concatenation_early')) and not(getattr(args.args_vision,'concatenation_early')):
                Ko = 1
        '''

    if hasattr(args,'args_embedding') and (len(vars(args.args_embedding))>0): #if not empty 
        if args.args_embedding.concatenation_early:
            Ko = Ko + args.args_embedding.embedding_dim

    if  hasattr(args,'args_vision') and (len(vars(args.args_vision))>0):   #if not empty 
        # Depend wether out_dim is implicit or defined by other parameters:
        if args.args_vision.concatenation_early:
            if hasattr(args.args_vision,'out_dim'):
                Ko = Ko + args.args_vision.out_dim
            else:
                vision_out_dim = args.args_vision.L*args.args_vision.h_dim//2
                Ko = Ko + vision_out_dim
    
    return(Ko)

def get_block_dims(args,Ko):
    blocks = args.blocks.copy()
    if Ko > 0:
        blocks[-1] = blocks[-1]*2
        if is_condition(args):
            blocks[-1][0] = 0
    blocks.append([args.out_dim])
    number_of_st_conv_blocks = len(blocks) - 3

    assert ((args.enable_padding)or((args.Kt - 1)*2*number_of_st_conv_blocks > args.L + 1)), f"The temporal dimension will decrease by {(args.Kt - 1)*2*number_of_st_conv_blocks} which doesn't work with initial dimension L: {args.L} \n you need to increase temporal dimension or add padding in STGCN_layer"
    return(blocks)

def get_gso_from_adj(dataset, args):
    # Compute Weighted Adjacency Matrix: 
    adj,n_vertex = load_adj(dataset, adj_type = args.adj_type, threshold=args.threshold)
    adj[adj < args.threshold] = 0
    adj = adj.to_numpy()
    gso = calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv':   
        gso = calc_chebynet_gso(gso)     # Calcul la valeur propre max du gso. Si lambda > 2 : gso = gso - I , sinon : gso = 2gso/lambda - I 
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    if args.single_station:
        gso = np.array([[1]]).astype(dtype=np.float32)
        n_vertex = 1
    gso = torch.from_numpy(gso).to(args.device)
    return gso,n_vertex


