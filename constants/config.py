import torch 
import argparse
import random
import torch.nn as nn
import os 

def get_config(model_name,learn_graph_structure = None,other_params =  {}):
    if model_name== 'CNN':
        config = dict(model_name= model_name,#epochs = [50],
                      enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',
                    c_in = 1, C_outs = [[16,2]],H_dims = [[16,16]], padding = 0
                    ) 

    if model_name== 'MTGNN':
        config = dict(model_name= model_name,#epochs = [30],
                    enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',   # VERIFIER CA  !!!!!!!!!!!!!!!!!!!!!!!
                    dilation_exponential=1,
                    c_in = 1,conv_channels=32, residual_channels=32, 
                    skip_channels=64, end_channels=128,layers=3,layer_norm_affline=True, 
                    gcn_true = True, # learn graph structure
                    buildA_true = True, # learn graph structure
                    gcn_depth = 2,propalpha=0.05,predefined_A=None,# pour la Graph Convolution
                    #Pour construction de matrice d'adjacence
                    subgraph_size=20,  #Dimension du sous-graph. A priori <= node_dim car issue de matrice générée depuis l'embedding des noeuds
                    node_dim=30,   #Dimension d'embedding. A priori <= num_nodes qui est définie dans utilities_DL.get_MultiModel_loss_args_emb_opts qui est la dimension d'embedding des noeuds
                    tanhalpha=3,
                    # ...

                    static_feat=None,  # Si = None, alors nodevec1 et nodevec2 sont  issues d'embedding different dans le graph constructor. Sinon Issue de static_feat qui est une matrice (statique pre-définie ?)
        )

    if (model_name == 'STGCN') or  (model_name == 'stgcn'):
        # Utilise la distance adjacency matrix 
        config = dict(model_name= model_name,#epochs = [100],
                        calib_prop = [0.5],
                        enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',
                        
                        Kt = 3, stblock_num=2,
                        act_fun=['glu'],#['glu','gtu'],
                        Ks =[2], #[3,2],
                        graph_conv_type = ['graph_conv'], # ['cheb_graph_conv', 'graph_conv'],
                        gso_type = ['sym_norm_lap'], # ['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'],
                        enable_bias = 'True',
                        adj_type = 'dist',
                        enable_padding = True,

                        threeshold = 0.3,gamma = 0.95,patience = 30
        )

    if model_name == 'DCRNN':
        config  = dict(model_name= model_name,
                       adj_type = 'dist',
                       cl_decay_steps = 1000,  # Tant que use_curriculum_learning = False, ne sera jamais utilisé
                       use_curriculum_learning = False, #Methode d'apprentissage. Pas besoin ici
                       input_dim = 1 ,  # Number of featuree :  Flow, Velocity ...
                       max_diffusion_step = 2, #1,2,3 ...
                       filter_type ='laplacian' , #'laplacian' # 'random_walk' , # 'dual_random_walk'
                       num_nodes = 1 , #1
                       num_rnn_layers = 1, #1
                       rnn_units = 1,
        )



    if model_name== 'LSTM':
        config = dict(model_name= model_name,#epochs = [30],
                        calib_prop = [0.5], 
                        enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',

                          h_dim =[16,32,64],C_outs = [[16,2],[32,2],[16,16,2]],num_layers = 2,bias = True,
                         bidirectional = [True,False]
        )
        
    if model_name == 'GRU':
        config = dict(model_name= model_name,#epochs = [30],
                        calib_prop = [0.5],
                        enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',

                         h_dim =[16,32,64],C_outs = [[16,2],[32,2],[16,16,2]],num_layers = 2,bias = True,
                         bidirectional = [True,False]
        )

    if model_name == 'RNN':
        config = dict(model_name= model_name,#epochs = [30], 
                       calib_prop = [0.5],
                        enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',

                        h_dim =[16,32,64],C_outs = [[16,2],[32,2],[16,16,2]],num_layers = 2,bias = True,
                         bidirectional = [True,False]
        )
        

    # === Common config for everyone: ===
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['optimizer'] = 'adamw' #['sgd','adam','adamw']
    config['weight_decay'] = 0.0005
    config['momentum'] = 0.99
    config['loss_function_type'] = 'quantile' # MSE
    config['single_station']= False
    config['batch_size'] = 32
    config['lr'] = 1e-3
    config['dropout'] = 0.2
    config['epochs'] = 100 if torch.cuda.is_available() else 2
    config['contextual_positions'] = {}


    # Optimization 
    if torch.cuda.is_available():
        config['num_workers'] = 2 # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
        config['persistent_workers'] = True # False 
        config['pin_memory'] = True # False 
        config['prefetch_factor'] = 2 # None, 2,3,4,5 ... 
        config['drop_last'] = False  # True
    else:
        config['num_workers'] = 0 # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
        config['persistent_workers'] = False # False 
        config['pin_memory'] = True # False 
        config['prefetch_factor'] = None # None, 2,3,4,5 ... 
        config['drop_last'] = False  # True      
    
    config['non_blocking'] = True
    config['mixed_precision'] = True # True # False
    config['torch_compile'] = False #True
    config['backend'] = 'inductor' #'cudagraphs'
    config['prefetch_all'] = False
    # ...

    # Scheduler 
    config['scheduler'] = True # None
    config['torch_scheduler_milestone'] = 40 # if scheduler == True 
    config['torch_scheduler_gamma'] = 0.99 # if scheduler == True 
    config['torch_scheduler_lr_start_factor'] = 0.1 # if scheduler == True 

    # === Ray config ===
    config['ray'] = False # True
    config['ray_scheduler'] = 'ASHA' #None
    config['ray_search_alg'] = None #  'HyperOpt'

    # Config Quantile Calibration 
    config['alpha'] = 0.1
    config['conformity_scores_type'] = 'max_residual'   # Define the function to compute the non-conformity scores
    config['quantile_method'] =  'compute_quantile_by_class' # 'classic' Define type of method used to calcul quantile.  'classic':  Quantile through the entiere dataset  / 'weekday_hour': 
    config['calibration_calendar_class'] = 0  # Calibrates data by defined calendar-class 

    # Config Time Embedding: 
    config['position'] = 'input'  # Position of time_embedding module : before or after the core model
    config['time_embedding'] = True
    config['type_calendar'] = 'tuple' # unique_long_embedding : embedding for a single long vector. tuple:  embedding of each element of the tuple
    config['calendar_class'] = 3
    config['specific_lr'] = False  #True, False
    config['embedding_dim'] = 3
    config['multi_embedding'] = False
    config['TE_transfer'] = False

    # Config DataSet:
    config['H'] = 6
    config['W'] = 1
    config['D'] = 1
    config['step_ahead'] = 1
    config['L'] = config['H']+config['W']+config['D']

    # Split proportion
    config['train_prop'] = 0.6
    config['calib_prop'] = 0.5
    config['valid_prop'] = 0.2  
    config['test_prop'] = 1 - (config['train_prop'] + config['valid_prop']) 
    assert config['train_prop']+ config['valid_prop'] < 1.0, f"train_prop + valid_prop = {config['train_prop']+ config['valid_prop']}. No Testing set"
    config['track_pi'] = False #True

    # Validation, K-fold
    config['validation'] = 'sliding_window'  # classic / sliding_window / 
    config['no_common_dates_between_set'] = False  #If True then a shift of dataset.shift_from_first_elmt is applied. Otherwise, some pattern could be within Training and Validation DataLoader
    config['K_fold'] = 1  # int. If 1 : classic validation (only 1 model), Else : validation with K_fold according 'config['validation']
    config['current_fold'] = 0

    # ===   ===
    config['abs_path'] =  ('/').join(f"{os.path.abspath(os.getcwd())}".split('/')[:-1]) + '/' # f"{os.path.abspath(os.getcwd())}/"

    # Add other parameters:
    for key in other_params.keys():
        config[key] = other_params[key]
    # ...

    # Define Output dimension: 
        if config['loss_function_type'] == 'MSE': out_dim = 1
        elif config['loss_function_type'] == 'quantile': out_dim = 2
        else: raise NotImplementedError(f"loss function {config['loss_function_type']} has not been implemented")
        config['out_dim'] = out_dim
    # ...

    # Vision Model:
    '''
    if 'vision_input_type' == 'image_per_stations': the model will extract feature by looking around a station
    if 'vision_input_type' == 'unique_image_through_lyon':  the model will extract feature by looking through the entiere map, and then return N outputs (one for each station)
    '''
    config['vision_input_type'] = 'unique_image_through_lyon' # 'image_per_stations' # 'unique_image_through_lyon'  

    return(config)
    

def optimizer_specific_lr(model,args):
    if args.model_name == 'CNN':
        if args.specific_lr:
            specific_lr = [{"params": model.Tembedding.parameters(), "lr": 1e-2},
                    {"params": model.Convs.parameters(), "lr": args.lr},
                    {"params": model.Dense_outs.parameters(), "lr": args.lr}
                ]
            
    elif args.model_name == 'STGCN':
        if args.specific_lr:
            specific_lr = [{"params": model.Tembedding.parameters(), "lr": 1e-2},
                    {"params": model.st_blocks.parameters(), "lr": args.lr},
                    {"params": model.output.parameters(), "lr": args.lr}
                    ]
    else:
        raise NotImplementedError(f'A specific lr by layer has been asked but it has not been defined for the model {args.model_name}.')
    
    return(specific_lr)


def get_config_embed(nb_words_embedding,embedding_dim,position):
    '''
    args
    -----
    nb_words_embedding : represent the number of expected class from tuple (weekday,hour,minute) 
    '''
    config_Tembed = dict(nb_words_embedding= nb_words_embedding,embedding_dim = embedding_dim, position=position)
    return(config_Tembed)

def get_args(model_name,learn_graph_structure = None,other_params =  {}):
    config = get_config(model_name,learn_graph_structure,other_params)
    args = get_parameters(config,description = None )
    return(args)

def get_parameters(config,description = None ):
    if description is None:
        description = config['model_name']
    parser = argparse.ArgumentParser(description=description)

    for key in config.keys():
        if type(config[key]) != list:
            default = config[key]
        else:
            ind = random.randint(0,len(config[key])-1)
            default = config[key][ind]
        parser.add_argument(f'--{key}', type=type(default), default=default)

    args = parser.parse_args(args=[])
    return(args)


def update_modif(args,name_gpu='cuda'):
    #Update modification:
    if args.loss_function_type == 'MSE': 
        args.out_dim = 1
        args.alpha = None
    elif args.loss_function_type == 'quantile': 
        args.out_dim = 2
        args.alpha = 0.1
    else: 
        raise NotImplementedError(f'loss function {args.loss_function_type} has not been implemented')
    #...
    
    # Modification according GPU availability: 
    if torch.cuda.is_available():
            args.device = name_gpu
            args.batch_size = 128
    else :
        args.device = 'cpu'
        args.batch_size = 16
    # ...

    args.L = args.W + args.D + args.H
    
    # Ray tuning function can't hundle with PyTorch multiprocessing : 
    if args.ray:
        args.num_workers = 0
        args.persistent_workers = False
    
    print(f"Model: {args.model_name}, K_fold = {args.K_fold}") 
    print(f"!!! Loss function: {args.loss_function_type} ")
    print("!!! Prediction sur une UNIQUE STATION et non pas les 40 ") if args.single_station else None
    return(args)


def update_args(args,subway_ds,dataset_names):

    # Update args according datasets choice: 
    args.dataset_names = dataset_names
    args.contextual_positions = subway_ds.contextual_positions

    if not 'subway_in' in dataset_names:
        args.L = 0

    if subway_ds.U_train.dim() == 3:
        args.C = 1
        args.N = subway_ds.U_train.size(1) 
    elif subway_ds.U_train.dim() == 4:
        args.C = subway_ds.U_train.size(1)
        args.N = subway_ds.U_train.size(2) 
    else:
        raise NotImplementedError("Feature vector like 'subway_ds.U_train' doesn't have the expected shape")
    # ...
    args.num_nodes = subway_ds.raw_values.size(1)

    return(args)



def display_config(args,args_embedding):
    # Args 
    optimizer = f"Optimizer: {args.optimizer}"
    lr = 'A specific LR by layer is used' if args.specific_lr else 'The same LR is used for each layer'
    calendar_class = f"Calendar class: {args.calendar_class}"
    quantile_method = f"Quantile Method: {args.quantile_method}"

    # Args Embedding 
    encoding = f"Encoding dimension: {args_embedding.nb_words_embedding}. Is related to Dictionnary size of the Temporal Embedding Layer \n " if args.time_embedding else '' 
    embedding_dim = f"Embedding dimension: {args_embedding.embedding_dim} \n " if args.time_embedding else '' 
    position = f"Position of the Embedding layer: {args_embedding.position}" if args.time_embedding else '' 
    print(f"Model : {args.model_name} \n {optimizer} \n {lr} \n {calendar_class} \n {quantile_method} \n {encoding} {embedding_dim} {position} ")
