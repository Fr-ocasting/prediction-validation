
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

def get_output_kernel_size(args):
     # Set Ko : Last Temporal Channel dimension before passing through output module :
    if args.enable_padding: 
        Ko = args.L  # if args.L > 0 else 1
    else :
        Ko = args.L - (args.Kt - 1) * 2 * args.stblock_num    

    if hasattr(args,'args_embedding') and (len(vars(args.args_embedding))>0): #if not empty 
        Ko = Ko + args.args_embedding.embedding_dim

    if  hasattr(args,'args_vision') and (len(vars(args.args_vision))>0):   #if not empty 
        # Depend wether out_dim is implicit or defined by other parameters:
        if hasattr(args.args_vision,'out_dim'):
            Ko = Ko + args.args_vision.out_dim
        else:
            vision_out_dim = args.args_vision.L*args.args_vision.h_dim//2
            Ko = Ko + vision_out_dim
    return(Ko)

def get_block_dims(args,Ko):
    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
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


