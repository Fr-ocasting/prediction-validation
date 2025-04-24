
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
    if  not (args.target_data in args.dataset_names):
        Ko = 0
        if is_condition(args):
            Ko = 1

    if hasattr(args,'args_embedding') and (len(vars(args.args_embedding))>0): #if not empty 
        if args.args_embedding.concatenation_early:
            Ko = Ko + args.args_embedding.embedding_dim

    if  hasattr(args,'args_vision') and (len(vars(args.args_vision))>0):   #if not empty 
        # Depend wether out_dim is implicit or defined by other parameters:
        if args.args_vision.concatenation_early:
            if False:
                if hasattr(args.args_vision,'out_dim'):
                    Ko = Ko + args.args_vision.out_dim
                else:
                    vision_out_dim = args.args_vision.L*args.args_vision.h_dim//2
                    Ko = Ko + vision_out_dim
            print('MODIFIER AUSSI ICI REMETTRE TRUE dl_models.STGCN.get_gso.py line 49')
            Ko = 14
    
    return(Ko)

def load_blocks(c_in,stblock_num,temporal_h_dim,spatial_h_dim,output_h_dim):
    blocks = []
    blocks.append([c_in])
    for l in range(stblock_num):
        blocks.append([temporal_h_dim, spatial_h_dim, temporal_h_dim])
    blocks.append([output_h_dim])
    return(blocks)


def get_block_dims(args,Ko):
    blocks = load_blocks(args.C, args.stblock_num,args.temporal_h_dim, args.spatial_h_dim,args.output_h_dim)
    #blocks = args.blocks.copy()
    if Ko > 0:
        blocks[-1] = blocks[-1]*2
        if  not(args.target_data in args.dataset_names) and is_condition(args):
            blocks[-1][0] = 0
    blocks.append([args.out_dim])
    number_of_st_conv_blocks = len(blocks) - 3

    assert ((args.enable_padding)or((args.Kt - 1)*2*number_of_st_conv_blocks > args.L + 1)), f"The temporal dimension will decrease by {(args.Kt - 1)*2*number_of_st_conv_blocks} which doesn't work with initial dimension L: {args.L} \n you need to increase temporal dimension or add padding in STGCN_layer"

    args.blocks = blocks
    return(args)

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


