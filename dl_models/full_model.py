from dl_models.time_embedding import TE_module
from dl_models.CNN_based_model import CNN
from dl_models.MTGNN import gtnet
from dl_models.RNN_based_model import RNN
from dl_models.STGCN import STGCN
from dl_models.STGCN_utilities import calc_chebynet_gso,calc_gso
from dl_models.dcrnn_model import DCRNNModel
from dl_models.vision_models.simple_feature_extractor import FeatureExtractor_ResNetInspired,MinimalFeatureExtractor,ImageAvgPooling

from load_adj import load_adj
import numpy as np 
import torch
import torch.nn as nn


import inspect

def filter_args(func, args):
    sig = inspect.signature(func)
    valid_args = {k: v for k, v in args.items() if k in sig.parameters}
    return valid_args


def load_vision_model(args_vision):
    if args_vision['model_name'] == 'ImageAvgPooling':
        filered_args = filter_args(ImageAvgPooling, args_vision)
        return ImageAvgPooling(**filered_args) 
    
    elif args_vision['model_name'] == 'MinimalFeatureExtractor':
        filered_args = filter_args(MinimalFeatureExtractor, args_vision)
        return MinimalFeatureExtractor(**filered_args)
    
    elif args_vision['model_name'] == 'FeatureExtractor_ResNetInspired':
        filered_args = filter_args(FeatureExtractor_ResNetInspired, args_vision)
        return FeatureExtractor_ResNetInspired(**filered_args)
    else:
        NotImplementedError(f"Model {args_vision['model_name']} has not been implemented")



class full_model(nn.Module):
    def __init__(self,args,args_embedding,dic_class2rpz,args_vision):
        super(full_model,self).__init__()

        # === Vision NetMob ===
        self.netmob_vision =  load_vision_model(args_vision) if 'netmob' in args.contextual_positions.keys() else None  

        # === TE ===
        self.te = TE_module(args,args_embedding,dic_class2rpz) if args.time_embedding else None

        # === Trafic Model ===
        self.core_model = load_model(args,args_embedding,dic_class2rpz,args_vision)


        # Add positions for each contextual data:
        if 'calendar' in args.contextual_positions.keys(): 
            self.pos_calendar = args.contextual_positions['calendar']
        if 'netmob' in args.contextual_positions.keys(): 
            self.pos_netmob = args.contextual_positions['netmob']

        if 'subway_in' in args.dataset_names :
            self.remove_trafic_inputs = False
        else:
            self.remove_trafic_inputs = True
            print('\nPREDICTION WILL BE BASED SOLELY ON CONTEXTUAL DATA !\n')
        # ...

    def forward(self,x,contextual = None):
        ''' 
        Args:
        -----
        x : 4-th order Tensor: Trafic Flow historical inputs [B,C,N,L]
        contextual : list of contextual data. 
            >>>> contextual[netmob_position]: [B,N,C,H,W,L]
            >>>> contextual[calendar]: [B]
        '''
        if self.remove_trafic_inputs:
            x = torch.Tensor().to(x)
        else:
            if x.dim() == 3:
                x = x.unsqueeze(1)


        # if NetMob data is on :
        if self.netmob_vision is not None: 

            # [B,N,C,H,W,L]
            netmob_video_batch = contextual[self.pos_netmob]
            B,N,C_netmob,H,W,L = netmob_video_batch.size()

            # Reshape:  [B,N,C,H,W,L] -> [B*N,C,H,W,L]
            netmob_video_batch = netmob_video_batch.reshape(B*N,C_netmob,H,W,L)

            # Forward : [B*N,C,H,W,L] ->  [B*N,Z] 
            extracted_feature = self.netmob_vision(netmob_video_batch)

            # Reshape  [B*N,Z] -> [B,C,N,Z]
            extracted_feature = extracted_feature.reshape(B,N,-1)
            extracted_feature = extracted_feature.unsqueeze(1)

            # Concat: [B,C,N,L],[B,C,N,Z] -> [B,C,N,L+Z]
            x = torch.cat([x,extracted_feature],dim = -1)
        # ...

        # if calendar data is on : 
        if self.te is not None:
            time_elt = contextual[self.pos_calendar].long()
            # Extract feature: [B] -> [B,C,N,L_calendar]
            time_elt = self.te(time_elt)
            
            # Concat: [B,C,N,L],[B,C,N,L_calendar] -> [B,C,N,L+L_calendar]
            x = torch.cat([x,time_elt],dim = -1)
        # ...


        # Core model 

        x = self.core_model(x)
        # ...

        return(x)


def load_model(args,args_embedding,dic_class2rpz,args_vision):
    if args.model_name == 'CNN': 
        model = CNN(args, kernel_size = (2,1),args_embedding = args_embedding,dic_class2rpz = dic_class2rpz)
    if args.model_name == 'MTGNN': 
        model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes, args.device, 
                    predefined_A=args.predefined_A, static_feat=args.static_feat, 
                    dropout=args.dropout, subgraph_size=args.subgraph_size, node_dim=args.node_dim, 
                    dilation_exponential=args.dilation_exponential, conv_channels=args.conv_channels, residual_channels=args.residual_channels, 
                    skip_channels=args.skip_channels, end_channels=args.end_channels, seq_length=args.L, in_dim=args.c_in, out_dim=args.out_dim, 
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=args.layer_norm_affline,args_embedding=args_embedding)
        
    if args.model_name == 'DCRNN':
        model_kwargs = vars(args)
        adj,num_nodes = load_adj(args.abs_path,adj_type = args.adj_type)
        model = DCRNNModel(adj, **model_kwargs)
        
    if args.model_name == 'STGCN':

        # Set Ko : Last Temporal Channel dimension before passing through output module :
        if args.enable_padding: 
            Ko = args.L  # if args.L > 0 else 1
        else :
            Ko = args.L - (args.Kt - 1) * 2 * args.stblock_num    

        # With padding, the output channel dimension will stay constant and equal to L
        # Sometimes, with no Trafic Data, L = 0, then we have to set Ko = 1, independant of L

        if args_embedding is not None:
            Ko = Ko + args_embedding.embedding_dim

        if args_vision is not None:
            Ko = Ko + args_vision['out_dim']
        #  ...

        # Define Blocks  (should be in a STGCN config file...)
        blocks = []
        blocks.append([1])
        for l in range(args.stblock_num):
            blocks.append([64, 16, 64])
        if Ko == 0:
            blocks.append([128])
        elif Ko > 0:
            blocks.append([128, 128])
        blocks.append([args.out_dim])
        # ...


        # Compute Weighted Adjacency Matrix: 
        adj,num_nodes = load_adj(args.abs_path,adj_type = args.adj_type)
        adj[adj < args.threeshold] = 0
        
        adj = adj.to_numpy()
        gso = calc_gso(adj, args.gso_type)
        if args.graph_conv_type == 'cheb_graph_conv':   
            gso = calc_chebynet_gso(gso)     # Calcul la valeur propre max du gso. Si lambda > 2 : gso = gso - I , sinon : gso = 2gso/lambda - I 
        gso = gso.toarray()
        gso = gso.astype(dtype=np.float32)
        if args.single_station:
            gso = np.array([[1]]).astype(dtype=np.float32)
            num_nodes = 1
        gso = torch.from_numpy(gso).to(args.device)
        # ...

        
        model = STGCN(args,gso, blocks,Ko, num_nodes,args_embedding = args_embedding,dic_class2rpz = dic_class2rpz,args_vision = args_vision).to(args.device)
        
        number_of_st_conv_blocks = len(blocks) - 3
        assert ((args.enable_padding)or((args.Kt - 1)*2*number_of_st_conv_blocks > args.L + 1)), f"The temporal dimension will decrease by {(args.Kt - 1)*2*number_of_st_conv_blocks} which doesn't work with initial dimension L: {args.L} \n you need to increase temporal dimension or add padding in STGCN_layer"

    if args.model_name == 'LSTM':
        model = RNN(args.L,args.h_dim,args.C_outs, args.num_layers,bias = args.bias,dropout = args.dropout,bidirectional = args.bidirectional,lstm = True)
    if args.model_name == 'GRU':
        model = RNN(args.L,args.h_dim,args.C_outs, args.num_layers,bias = args.bias,dropout = args.dropout,bidirectional = args.bidirectional, gru = True)
    if args.model_name == 'RNN':
        model = RNN(args.L,args.h_dim,args.C_outs, args.num_layers, nonlinearity = 'tanh',bias = args.bias,dropout = args.dropout,bidirectional = args.bidirectional) 
    return(model)