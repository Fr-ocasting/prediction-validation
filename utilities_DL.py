import numpy as np
import torch 
import torch.nn as nn 
from torch.optim import SGD,Adam,AdamW

# Personnal import: 
from load_adj import load_adj
from config import optimizer_specific_lr, get_config_embed, get_parameters,display_config
from calendar_class import get_time_slots_labels
from load_DataSet_subway_15 import load_normalized_dataset
from DL_class import DictDataLoader,QuantileLoss

# Models : 
from dl_models.CNN_based_model import CNN
from dl_models.MTGNN import gtnet
from dl_models.RNN_based_model import RNN
from dl_models.STGCN import STGCNChebGraphConv, STGCNGraphConv
from dl_models.STGCN_utilities import calc_chebynet_gso,calc_gso


def load_all(subway_in,args,time_step_per_hour,step_ahead,H,D,W,invalid_dates,
             embedding_dim=2,position = 'input',single_station = False):
    ''' Load dataset, dataloader, loss function, Model, Optimizer, Trainer '''
    df = subway_in[['Ampère Victor Hugo']] if single_station else subway_in
    dataset,data_loader,dic_class2rpz,dic_rpz2class,nb_words_embedding = data_generator(df,args,time_step_per_hour,step_ahead,H,D,W,invalid_dates)

    # Time Embedding Config
    config_Tembed = get_config_embed(nb_words_embedding = nb_words_embedding,embedding_dim = embedding_dim,position = position)
    args_embedding = get_parameters(config_Tembed,description = 'TimeEmbedding')
    # Print config :
    display_config(args,args_embedding)
    # Quantile Loss
    loss_function = get_loss(args.loss_function_type,args)

    # Load Model
    model = load_model(args,args_embedding).to(args.device)

    # Config optimizer:
    optimizer = choose_optimizer(model,args)

    return(dataset,data_loader,dic_class2rpz,dic_rpz2class,args_embedding,loss_function,model,optimizer)



def get_dic_results(trainer,pi):
    results = {}
    results['MPIW'] = pi.mpiw
    results['PICP'] = pi.picp  
    results['Q'] = pi.Q_tensor.mean().item()
    results['last train loss'] = trainer.train_loss[-1]
    results['last valid loss'] = trainer.valid_loss[-1] 
    return(results)


def data_generator(df,args,time_step_per_hour,step_ahead,H,D,W,invalid_dates):
    (dataset,U,Utarget,remaining_dates) = load_normalized_dataset(df,time_step_per_hour,args.train_prop,step_ahead,H,D,W,invalid_dates)
    print(f"{len(df.columns)} nodes (stations) have been considered. \n ")
    time_slots_labels,dic_class2rpz,dic_rpz2class,nb_words_embedding = get_time_slots_labels(dataset,type_class= args.calendar_class)
    data_loader_obj = DictDataLoader(U,Utarget,args.train_prop,args.valid_prop,validation = 'classic', shuffle = True, calib_prop=args.calib_prop, time_slots = time_slots_labels)
    data_loader = data_loader_obj.get_dictdataloader(args.batch_size)


    # Print Information
    training_set =  f"between {str(remaining_dates.iloc[0].item())} and {str(remaining_dates.iloc[int(len(remaining_dates)*args.train_prop)].item())} \
        Contains {int(len(remaining_dates)*args.train_prop)} sequences by spatial unit"
    validation_set = f"Doesn't exist" if (args.valid_prop == 0) else f"between {str(remaining_dates.iloc[int(len(remaining_dates)*args.train_prop)].item())} and {str(remaining_dates.iloc[int(len(remaining_dates)*(args.train_prop+args.valid_prop))].item())}.\
        Contains {int(len(remaining_dates)*args.valid_prop)} sequences by spatial unit"
    testing_set = f"Doesn't exist" if ((args.valid_prop + args.train_prop == 1) or (args.train_prop == 1 )) else f"between {str(remaining_dates.iloc[int(len(remaining_dates)*(args.train_prop+args.valid_prop))].item())} and {str(remaining_dates.iloc[-1].item())}.\
        Contains {int(len(remaining_dates)*(1-args.valid_prop-args.train_prop))} sequences by spatial unit"
    
    print(f"Initial size of the data: {len(dataset.df)}. \
      \nNumber of forbidden dates: {len(invalid_dates)} which can't be present in any sequence . \
      \nProportion of remaining data: {'{:.0%}'.format(len(remaining_dates)/len(dataset.df))} \n \
      \nTrain set {training_set} \
      \nValid set {validation_set}  \
      \nTest set {testing_set} \n  \
      ")
    # ...
    
    return(dataset,data_loader,dic_class2rpz,dic_rpz2class,nb_words_embedding)

def get_loss(loss_function_type,args = None):
    if (loss_function_type == 'mse') or  (loss_function_type == 'MSE') or (loss_function_type == 'Mse'):
        return nn.MSELoss()
    if (loss_function_type == 'quantile') or (loss_function_type == 'Quantile') (loss_function_type == 'quantile loss') or (loss_function_type == 'QunatileLoss') or (loss_function_type == 'Qunatile Loss'):
        quantiles = torch.Tensor([args.alpha/2,1-args.alpha/2]).to(args.device)
        assert args.out_dim == len(quantiles), f"Output dimension {args.out_dim} doesn't match with the number of estimated quantiles {len(quantiles)}"
        return QuantileLoss(quantiles)

def choose_optimizer(model,args):
    # Training and Calibration :
    specific_lr = optimizer_specific_lr(model,args) if args.specific_lr else None

    if args.optimizer == 'adam':
        if specific_lr is not None: 
            return Adam(specific_lr,lr=args.lr,weight_decay= args.weight_decay)
        else:
            return Adam(model.parameters(),lr=args.lr,weight_decay= args.weight_decay)
    elif args.optimizer == 'sgd':
        if specific_lr is not None: 
            return SGD(specific_lr,lr=args.lr,weight_decay=args.weight_decay, momentum = args.momentum)
        else:
            return SGD(model.parameters(),lr=args.lr,weight_decay =args.weight_decay, momentum = args.momentum)
    elif args.optimizer == "adamw":
        if specific_lr is not None: 
            return AdamW(specific_lr,lr=args.lr,weight_decay= args.weight_decay)
        else:
            return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else :
        raise NotImplementedError(f'ERROR: The optimizer is not set in args or is not implemented.')


def load_model(args,args_embedding):
    if args.model_name == 'CNN': 
        model = CNN(args.c_in, args.H_dims, args.C_outs, kernel_size = (2,), L=args.seq_length, padding = args.padding,dropout = args.dropout,args_embedding = args_embedding)
    if args.model_name == 'MTGNN': 
        model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes, args.device, 
                    predefined_A=args.predefined_A, static_feat=args.static_feat, 
                    dropout=args.dropout, subgraph_size=args.subgraph_size, node_dim=args.node_dim, 
                    dilation_exponential=args.dilation_exponential, conv_channels=args.conv_channels, residual_channels=args.residual_channels, 
                    skip_channels=args.skip_channels, end_channels=args.end_channels, seq_length=args.seq_length, in_dim=args.c_in, out_dim=args.out_dim, 
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=args.layer_norm_affline)
        
    if args.model_name == 'STGCN':
        Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
        if args.enable_padding:
            Ko = args.n_his
        blocks = []
        blocks.append([1])
        for l in range(args.stblock_num):
            blocks.append([64, 16, 64])
        if Ko == 0:
            blocks.append([128])
        elif Ko > 0:
            blocks.append([128, 128])
        blocks.append([args.out_dim])

        # Intégrer les deux fonction calc_gso et calc_chebynet_gso. Regarder comment est représenté l'input.
        adj,num_nodes = load_adj(adj_type = args.adj_type)
        adj[adj < args.threeshold] = 0
        
        adj = adj.to_numpy()
        gso = calc_gso(adj, args.gso_type)
        if args.graph_conv_type == 'cheb_graph_conv':   
            gso = calc_chebynet_gso(gso)     # Calcul la valeur propre max du gso. Si lambda > 2 : gso = gso - I , sinon : gso = 2gso/lambda - I 
        gso = gso.toarray()
        gso = gso.astype(dtype=np.float32)
        args.gso = torch.from_numpy(gso).to(args.device)

        if args.graph_conv_type == 'cheb_graph_conv':
            model = STGCNChebGraphConv(args, blocks, args.num_nodes,args_embedding = args_embedding).to(args.device)
        else:
            model = STGCNGraphConv(args, blocks, args.num_nodes,args_embedding = args_embedding).to(args.device)
        number_of_st_conv_blocks = len(blocks) - 3
        assert ((args.enable_padding)or((args.Kt - 1)*2*number_of_st_conv_blocks > args.seq_length + 1)), f"The temporal dimension will decrease by {(args.Kt - 1)*2*number_of_st_conv_blocks} which doesn't work with initial dimension L: {args.seq_length} \n you need to increase temporal dimension or add padding in STGCN_layer"
        print(f"Ko: {Ko}, enable padding: {args.enable_padding}")
    if args.model_name == 'LSTM':
        model = RNN(args.seq_length,args.h_dim,args.C_outs, args.num_layers,bias = args.bias,dropout = args.dropout,bidirectional = args.bidirectional,lstm = True)
        # nn.LSTM(input_size = c_in, hidden_size = h_dim, num_layers=num_layers,bias=bias,batch_first=batch_first,dropout=dropout,bidirectional=bidirectional)
    if args.model_name == 'GRU':
        model = RNN(args.seq_length,args.h_dim,args.C_outs, args.num_layers,bias = args.bias,dropout = args.dropout,bidirectional = args.bidirectional, gru = True)
        #self.rnn = nn.GRU(input_size = c_in, hidden_size = h_dim, num_layers=num_layers,bias=bias,batch_first=batch_first,dropout=dropout,bidirectional=bidirectional)
    if args.model_name == 'RNN':
        model = RNN(args.seq_length,args.h_dim,args.C_outs, args.num_layers, nonlinearity = 'tanh',bias = args.bias,dropout = args.dropout,bidirectional = args.bidirectional)
        #self.rnn = nn.RNN(input_size = c_in, hidden_size = h_dim, num_layers=num_layers,nonlinearity=nonlinearity,bias=bias,batch_first=batch_first,dropout=dropout,bidirectional=bidirectional) 
    return(model)