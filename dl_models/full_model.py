from dl_models.time_embedding import TE_module
from dl_models.CNN_based_model import CNN
from dl_models.MTGNN import gtnet
from dl_models.RNN_based_model import RNN
from dl_models.STGCN import STGCNChebGraphConv, STGCNGraphConv
from dl_models.STGCN_utilities import calc_chebynet_gso,calc_gso
from dl_models.dcrnn_model import DCRNNModel

from load_adj import load_adj
import numpy as np 
import torch
import torch.nn as nn


class full_model(nn.Module):
    def __init__(self,args,args_embedding,dic_class2rpz):
        super(full_model,self).__init__()
        self.core_model = load_model(args,args_embedding,dic_class2rpz)
        self.te = TE_module(args,args_embedding,dic_class2rpz) if args.time_embedding else None

        # Add positions for each contextual data:
        if 'calendar' in args.contextual_positions.keys(): 
            self.pos_calendar = args.contextual_positions['calendar']
        if 'netmob' in args.contextual_positions.keys(): 
            self.pos_netmob = args.contextual_positions['netmob']
        # ...

    def forward(self,x,contextual = None):
        # if calendar data is on : 
        if self.te is not None:
            time_elt = contextual[self.pos_calendar].long()
            x = self.te(x,time_elt)
        # ...

        # Core model 
        x = self.core_model(x)
        # ...

        return(x)


def load_model(args,args_embedding,dic_class2rpz):
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
        Ko = args.L - (args.Kt - 1) * 2 * args.stblock_num
        if args.enable_padding:
            Ko = args.L
        if args_embedding is not None:
            Ko = Ko + args_embedding.embedding_dim
        blocks = []
        blocks.append([1])
        for l in range(args.stblock_num):
            blocks.append([64, 16, 64])
        if Ko == 0:
            blocks.append([128])
        elif Ko > 0:
            blocks.append([128, 128])
        blocks.append([args.out_dim])

        #print(f"Ko: {Ko}, enable padding: {args.enable_padding}")
        #print(f'Blocks: {blocks}')
        # Intégrer les deux fonction calc_gso et calc_chebynet_gso. Regarder comment est représenté l'input.
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

        if args.graph_conv_type == 'cheb_graph_conv':
            model = STGCNChebGraphConv(args,gso, blocks, num_nodes,args_embedding = args_embedding,dic_class2rpz = dic_class2rpz).to(args.device)
        else:
            model = STGCNGraphConv(args,gso, blocks, num_nodes,args_embedding = args_embedding,dic_class2rpz = dic_class2rpz).to(args.device)
        
        number_of_st_conv_blocks = len(blocks) - 3
        assert ((args.enable_padding)or((args.Kt - 1)*2*number_of_st_conv_blocks > args.L + 1)), f"The temporal dimension will decrease by {(args.Kt - 1)*2*number_of_st_conv_blocks} which doesn't work with initial dimension L: {args.L} \n you need to increase temporal dimension or add padding in STGCN_layer"

    if args.model_name == 'LSTM':
        model = RNN(args.L,args.h_dim,args.C_outs, args.num_layers,bias = args.bias,dropout = args.dropout,bidirectional = args.bidirectional,lstm = True)
    if args.model_name == 'GRU':
        model = RNN(args.L,args.h_dim,args.C_outs, args.num_layers,bias = args.bias,dropout = args.dropout,bidirectional = args.bidirectional, gru = True)
    if args.model_name == 'RNN':
        model = RNN(args.L,args.h_dim,args.C_outs, args.num_layers, nonlinearity = 'tanh',bias = args.bias,dropout = args.dropout,bidirectional = args.bidirectional) 
    return(model)