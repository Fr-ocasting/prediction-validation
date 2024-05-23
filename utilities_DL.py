import numpy as np
import torch 
import torch.nn as nn 
import pandas as pd
from datetime import datetime, timedelta
from torch.optim import SGD,Adam,AdamW
from torch.optim.lr_scheduler import LinearLR,ExponentialLR,SequentialLR
import os 

# Personnal import: 
from load_adj import load_adj
from config import optimizer_specific_lr, get_config_embed, get_parameters,display_config
from calendar_class import get_time_slots_labels
from load_DataSet import load_normalized_dataset
from DL_class import DictDataLoader,QuantileLoss,DataSet
from TE_transfer_learning import TE_transfer
from save_results import Dataset_get_save_folder, read_object, save_object 

# Models : 
from dl_models.CNN_based_model import CNN
from dl_models.MTGNN import gtnet
from dl_models.RNN_based_model import RNN
from dl_models.STGCN import STGCNChebGraphConv, STGCNGraphConv
from dl_models.STGCN_utilities import calc_chebynet_gso,calc_gso
from dl_models.time_embedding import TE_adder
from dl_models.dcrnn_model import DCRNNModel
# Load Loss 
def get_MultiModel_loss_args_emb_opts(args,nb_words_embedding,dic_class2rpz,n_vertex = None):
    args.num_nodes = n_vertex
    loss_function = get_loss(args.loss_function_type,args)

    if args.time_embedding:
        config_Tembed = get_config_embed(nb_words_embedding,embedding_dim = args.embedding_dim,position = args.position)
        args_embedding = get_parameters(config_Tembed,description = 'TimeEmbedding')
    else:
        args_embedding = None

    model_opt_sched_list = [load_model_and_optimizer(args,args_embedding,dic_class2rpz) for _ in range(args.K_fold)]
    Model_list = [model_opt[0] for model_opt in model_opt_sched_list]
    if args.TE_transfer:
        if os.path.exists(f'{args.abs_path}data/Trained_Time_Embedding{args.embedding_dim}.pkl'):
            Model_list = [TE_transfer(model,n_vertex,args,model_dir = 'data/') for model in Model_list]
        else:
            print('TE impossible')
                                      
        
    Optimizer_list = [model_opt[1] for model_opt in model_opt_sched_list]
    Scheduler_list = [model_opt[2] for model_opt in model_opt_sched_list]
    return(loss_function,Model_list,Optimizer_list,Scheduler_list,args_embedding)


def get_associated__df_verif_index(dataset,date,iloc):
    mask = (dataset.df_verif.iloc[:,iloc] == date)
    try:
        associated_index = dataset.df_verif[mask].index.item()
    except:
        associated_index = None
    return(associated_index)


def find_nearest_date(date_series, date, inferior=True):
    """
    Find the nearest index of the timestamp <= or >= 'date' in 'date_series'.
    Parameters:
        date_series (pd.Series): A series of timestamps.
        date (pd.Timestamp): The reference timestamp.
        inferior (bool): If True, search for the nearest date <= 'date', else >= 'date'.
    Returns:
        int or None: The index of the nearest date, or None if not found.
    """
    # Calculating the difference
    diff = date_series - date
    
    if inferior:
        # Filtering to get the nearest <= date
        filtered_series = diff[diff <= pd.Timedelta(0)]
        if not filtered_series.empty:
            nearest_index = filtered_series.idxmax()
        else:
            return None,None
    else:
        # Filtering to get the nearest >= date
        filtered_series = diff[diff >= pd.Timedelta(0)]
        if not filtered_series.empty:
            nearest_index = filtered_series.idxmin()
        else:
            return None,None
        
    nearest_indice = date_series.index.get_loc(nearest_index)
    
    return nearest_index,nearest_indice

def load_model_and_optimizer(args,args_embedding,dic_class2rpz):
    model = load_model(args,args_embedding,dic_class2rpz).to(args.device)
    # Config optimizer:
    optimizer = choose_optimizer(model,args)
    scheduler = load_scheduler(optimizer,args)
    return(model,optimizer,scheduler)

def get_DataSet_and_invalid_dates(abs_path,folder_path,file_name,W,D,H,step_ahead,single_station = False):
    df,invalid_dates,time_step_per_hour = load_raw_data(abs_path,folder_path,file_name,single_station = single_station)
    dataset = DataSet(df,time_step_per_hour=time_step_per_hour, Weeks = W, Days = D, historical_len= H,step_ahead=step_ahead)
    return(dataset,invalid_dates)

def load_raw_data(abs_path,folder_path,file_name,single_station = False,):
    if (file_name == 'preprocessed_subway_15_min.csv') | (file_name == 'subway_IN_interpol_neg_15_min_2019_2020.csv') | (file_name=='subway_IN_interpol_neg_15_min_16Mar2019_1Jun2020.csv'):
        subway_in = pd.read_csv(abs_path + folder_path+file_name,index_col = 0)
        subway_in.columns.name = 'Station'
        subway_in.index = pd.to_datetime(subway_in.index)

        subway_in = subway_in[['Ampère Victor Hugo']] if single_station else subway_in

        # Define Invalid Dates : 
        list_of_invalid_period = []
        list_of_invalid_period.append([datetime(2019,1,10,15,30),datetime(2019,1,14,15,30)])
        list_of_invalid_period.append([datetime(2019,1,30,8,15),datetime(2019,1,30,10,30)])
        list_of_invalid_period.append([datetime(2019,2,18,11),datetime(2019,2,18,13)])
        list_of_invalid_period.append([datetime(2019,4,23,14),datetime(2019,4,28,14)])
        list_of_invalid_period.append([datetime(2019,6,26,11),datetime(2019,6,28,4)])
        list_of_invalid_period.append([datetime(2019,10,27),datetime(2019,10,28,16)])
        list_of_invalid_period.append([datetime(2019,12,21,15,45),datetime(2019,12,21,16,45)])

        invalid_dates = []
        time_step_per_hour = (60*60)/(subway_in.iloc[1].name - subway_in.iloc[0].name).seconds
        for start,end in list_of_invalid_period:
            invalid_dates = invalid_dates + list(pd.date_range(start,end,freq = f'{60/time_step_per_hour}min'))

        # Restrain invalid_dates to the df: 
        invalid_dates = list(set(invalid_dates) & set(subway_in.index))

        print(f"coverage period: {subway_in.index.min()} - {subway_in.index.max()}")
        print(f"Time-step per hour: {time_step_per_hour}")

    
    else:
        raise NotImplementedError(f"file name option '{file_name}' has not been defined in 'load_row_data' ")
        
    return(subway_in,invalid_dates,time_step_per_hour)

def load_prediction(trainer,dataset,dataloader,args,training_mode,normalize):
    data=  [[x,y,t] for x,y,t in dataloader[training_mode]] 
    X = torch.cat([x for x,_,_ in data]).to(args.device)
    Y = torch.cat([y for _,y,_ in data]).to(args.device)
    T = torch.cat([t for _,_,t in data]).to(args.device)
    if normalize:
        Preds,Y,T = trainer.testing(dataset,False,training_mode,X, Y,T)
    else :
        Preds,Y,T = trainer.test_prediction(False,training_mode,X,Y,T)
    return(Preds,Y,T)


def get_dic_results(trainer,pi):
    results = {}
    results['MPIW'] = pi.mpiw
    results['PICP'] = pi.picp  
    results['Q'] = pi.Q_tensor.mean().item()
    results['last train loss'] = trainer.train_loss[-1]
    results['last valid loss'] = trainer.valid_loss[-1] 
    return(results)

def display_info_on_dataset(dataset,remaining_dates,train_indice = None,valid_indice = None):
    assert train_indice is not None, 'Train / Valid Split Indices has not been defined'

    if len(remaining_dates) > 0:
        train_dates1 =  f"{str(remaining_dates.iloc[0].item())}" 
        train_dates2 = f"{str(remaining_dates.iloc[train_indice].item())}"
        len_train = f"{len(remaining_dates[:train_indice])}"

        valid_dates1 =  None if (dataset.valid_prop == 0) else f"{str(remaining_dates.iloc[train_indice].item())}"
        valid_dates2 =  None if (dataset.valid_prop == 0) else f"{str(remaining_dates.iloc[valid_indice].item())}"  
        len_valid = f"{len(remaining_dates[train_indice:valid_indice])}"      

        test_dates1 =  None if ((abs(dataset.valid_prop + dataset.train_prop -1 ) < 1e-4) or (dataset.train_prop == 1 )) else f"{str(remaining_dates.iloc[valid_indice].item())}"
        test_dates2 =  None if ((abs(dataset.valid_prop + dataset.train_prop -1 ) < 1e-4) or (dataset.train_prop == 1 )) else f"{str(remaining_dates.iloc[-1].item())}"  
        len_test = f"{len(remaining_dates[valid_indice:])}"        

    else:
        train_dates1,train_dates2,valid_dates1,valid_dates2,test_dates1,test_dates2 = f"      None      ",f"      None      ",f"      None      ",f"      None      ",f"      None      ",f"      None      "
    

    print(f"Length full df: {len(dataset.df)}. \
    \n{'{:.0%}'.format(len(dataset.df_verif)/len(dataset.df))} of remaining dates after shifting lagged feature\
    \n{'{:.0%}'.format(1-len(remaining_dates)/len(dataset.df_verif))} of forbidden dates among remaining dates. \
    \n{'{:.0%}'.format(len(remaining_dates)/len(dataset.df))} of remaining sequences from Initial DataFrame \n " )

    print('              -- First Train   --   Last Train - First valid    --    Last Valid - First Test    --  Last Test')
    print(f"            {train_dates1}   --   {train_dates2 if train_dates2 == valid_dates1 else train_dates2 + ' None'}      --     {valid_dates2 if valid_dates2 == test_dates1 else valid_dates2 + ' None'}    --  {test_dates2}" ) 
    print(f" Len Train/Valid/Test:           {len_train}            ---            {len_valid}             ---             {len_test}\n")
    # ...

def get_loss(loss_function_type,args = None):
    if (loss_function_type == 'mse') or  (loss_function_type == 'MSE') or (loss_function_type == 'Mse'):
        return nn.MSELoss()
    if (loss_function_type == 'quantile') or (loss_function_type == 'Quantile') (loss_function_type == 'quantile loss') or (loss_function_type == 'QunatileLoss') or (loss_function_type == 'Qunatile Loss'):
        quantiles = torch.Tensor([args.alpha/2,1-args.alpha/2]).to(args.device)
        assert args.out_dim == len(quantiles), f"Output dimension {args.out_dim} doesn't match with the number of estimated quantiles {len(quantiles)}"
        return QuantileLoss(quantiles)

def choose_optimizer(model,args):
    # Training and Calibration :
    specific_lr = optimizer_specific_lr(model,args) if ((args.specific_lr) and (args.time_embedding)) else None

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
    
def load_scheduler(optimizer,args):
    if args.scheduler is None :
        scheduler = None
    else:
        scheduler1 = LinearLR(optimizer,total_iters = args.torch_scheduler_milestone, start_factor = args.torch_scheduler_lr_start_factor)
        scheduler2 = ExponentialLR(optimizer, gamma=args.torch_scheduler_gamma)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[args.torch_scheduler_milestone])
    return(scheduler)

def load_init_trainer(folder_path,file_name,args):
    # Load dataset and invalid_dates 
    dataset,invalid_dates = get_DataSet_and_invalid_dates(args.abs_path,folder_path,file_name,args.W,args.D,args.H,args.step_ahead,single_station = args.single_station)
    (Datasets,DataLoader_list,time_slots_labels,dic_class2rpz,dic_rpz2class,nb_words_embedding) = dataset.split_K_fold(args,invalid_dates)
    return(Datasets,DataLoader_list,dic_class2rpz,nb_words_embedding,time_slots_labels,dic_rpz2class)


def load_model(args,args_embedding,dic_class2rpz):
    if args.model_name == 'CNN': 
        model = CNN(args, kernel_size = (2,),args_embedding = args_embedding,dic_class2rpz = dic_class2rpz)
    if args.model_name == 'MTGNN': 
        model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes, args.device, 
                    predefined_A=args.predefined_A, static_feat=args.static_feat, 
                    dropout=args.dropout, subgraph_size=args.subgraph_size, node_dim=args.node_dim, 
                    dilation_exponential=args.dilation_exponential, conv_channels=args.conv_channels, residual_channels=args.residual_channels, 
                    skip_channels=args.skip_channels, end_channels=args.end_channels, seq_length=args.L, in_dim=args.c_in, out_dim=args.out_dim, 
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=args.layer_norm_affline,args_embedding=args_embedding)
        model = TE_adder(model,args,args_embedding,dic_class2rpz)
    if args.model_name == 'DCRNN':
        model_kwargs = vars(args)
        adj,num_nodes = load_adj(args.abs_path,adj_type = args.adj_type)
        model = DCRNNModel(adj, **model_kwargs)
        model = TE_adder(model,args,args_embedding,dic_class2rpz)
        
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
        args.gso = torch.from_numpy(gso).to(args.device)

        if args.graph_conv_type == 'cheb_graph_conv':
            model = STGCNChebGraphConv(args, blocks, num_nodes,args_embedding = args_embedding,dic_class2rpz = dic_class2rpz).to(args.device)
        else:
            model = STGCNGraphConv(args, blocks, num_nodes,args_embedding = args_embedding,dic_class2rpz = dic_class2rpz).to(args.device)
        
        model = TE_adder(model,args,args_embedding,dic_class2rpz)
            #model = STGCNGraphConv(args, blocks, num_nodes,args_embedding = args_embedding,dic_class2rpz = dic_class2rpz).to(args.device)
        number_of_st_conv_blocks = len(blocks) - 3
        assert ((args.enable_padding)or((args.Kt - 1)*2*number_of_st_conv_blocks > args.L + 1)), f"The temporal dimension will decrease by {(args.Kt - 1)*2*number_of_st_conv_blocks} which doesn't work with initial dimension L: {args.L} \n you need to increase temporal dimension or add padding in STGCN_layer"

    if args.model_name == 'LSTM':
        model = RNN(args.L,args.h_dim,args.C_outs, args.num_layers,bias = args.bias,dropout = args.dropout,bidirectional = args.bidirectional,lstm = True)
        # nn.LSTM(input_size = c_in, hidden_size = h_dim, num_layers=num_layers,bias=bias,batch_first=batch_first,dropout=dropout,bidirectional=bidirectional)
    if args.model_name == 'GRU':
        model = RNN(args.L,args.h_dim,args.C_outs, args.num_layers,bias = args.bias,dropout = args.dropout,bidirectional = args.bidirectional, gru = True)
        #self.rnn = nn.GRU(input_size = c_in, hidden_size = h_dim, num_layers=num_layers,bias=bias,batch_first=batch_first,dropout=dropout,bidirectional=bidirectional)
    if args.model_name == 'RNN':
        model = RNN(args.L,args.h_dim,args.C_outs, args.num_layers, nonlinearity = 'tanh',bias = args.bias,dropout = args.dropout,bidirectional = args.bidirectional)
        #self.rnn = nn.RNN(input_size = c_in, hidden_size = h_dim, num_layers=num_layers,nonlinearity=nonlinearity,bias=bias,batch_first=batch_first,dropout=dropout,bidirectional=bidirectional) 
    return(model)




# =========== Surement à supprimer : =============

def data_generator(df,args,time_step_per_hour,step_ahead,H,D,W,invalid_dates):
    (dataset,U,Utarget,remaining_dates) = load_normalized_dataset(df,time_step_per_hour,args.train_prop,args.valid_prop,step_ahead,H,D,W,invalid_dates)
    print(f"{len(df.columns)} nodes (stations) have been considered. \n ")
    time_slots_labels,dic_class2rpz,dic_rpz2class,nb_words_embedding = get_time_slots_labels(dataset,type_class= args.calendar_class,type_calendar = args.type_calendar)
    data_loader_obj = DictDataLoader(U,Utarget,args.train_prop,args.valid_prop,validation = args.validation, shuffle = True, calib_prop=args.calib_prop, time_slots = time_slots_labels)
    data_loader = data_loader_obj.get_dictdataloader(args.batch_size)
    # Print Information
    _,train_ind = find_nearest_date(remaining_dates.iloc[:,0],dataset.last_date_train,inferior = True)
    _,valid_ind = find_nearest_date(remaining_dates.iloc[:,0],dataset.last_date_valid,inferior = True)
    display_info_on_dataset(dataset,remaining_dates,train_ind,valid_ind)

    return(dataset,data_loader,dic_class2rpz,dic_rpz2class,nb_words_embedding)


def load_all(abs_path,folder_path,file_name,args,step_ahead,H,D,W,
             embedding_dim=2,position = 'input',single_station = False):
    ''' Load dataset, dataloader, loss function, Model, Optimizer, Trainer '''
    df,invalid_dates,time_step_per_hour = load_raw_data(abs_path,folder_path,file_name,single_station = False)

    dataset,data_loader,dic_class2rpz,dic_rpz2class,nb_words_embedding = data_generator(df,args,time_step_per_hour,step_ahead,H,D,W,invalid_dates)

    # Time Embedding Config
    config_Tembed = get_config_embed(nb_words_embedding = nb_words_embedding,embedding_dim = embedding_dim,position = position)
    args_embedding = get_parameters(config_Tembed,description = 'TimeEmbedding') if args.time_embedding else None
    # Print config :
    display_config(args,args_embedding)

    # Quantile Loss
    loss_function = get_loss(args.loss_function_type,args)

    # Load Model
    if type(data_loader) == list:
        model,optimizer,scheduler = [],[],[]
        for i in range(len(data_loader)):
            mod,opt,sched = load_model_and_optimizer(args,args_embedding,dic_class2rpz)
            model.append(mod)
            optimizer.append(opt)
            scheduler.append(sched)
    else:
        model,optimizer,scheduler = load_model_and_optimizer(args,args_embedding,dic_class2rpz)

    return(dataset,data_loader,dic_class2rpz,dic_rpz2class,args_embedding,loss_function,model,optimizer,invalid_dates,scheduler)
