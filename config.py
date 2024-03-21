import torch 
import argparse
import random
import torch.nn as nn

def get_config(model_name,learn_graph_structure = None,other_params =  {}):
    if model_name== 'CNN':
        config = dict(model_name= model_name,epochs = [50], lr = [1e-4],batch_size = [32],
                      dropout = [0.2],enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',
                      scheduler = None,ray = False,
                    c_in = 1, C_outs = [[16,2]],H_dims = [[16,16]],out_dim = 2, padding = 0
                    ) 

    if model_name== 'MTGNN':
        config = dict(model_name= model_name,epochs = [30], lr = [1e-4], batch_size = [64],dropout = [0.2],
                    enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',

                    gcn_true = False, buildA_true = False, gcn_depth = 2,propalpha=0.05,predefined_A=None,# inutile ici car pas de Graph Convolution
                    subgraph_size=20,node_dim=40,tanhalpha=3,static_feat=None,  # inutile aussi, c'est pour la construction de matrice d'adjacence

                    num_nodes = 40,dilation_exponential=1,
                    
                    c_in = 1,conv_channels=32, residual_channels=32, 
                    skip_channels=64, end_channels=128,out_dim=2,layers=3,layer_norm_affline=True, 
                    scheduler = None, ray = False
                    )
        if learn_graph_structure is not None:
            config['gcn_true'],config['buildA_true'] = True,True   

    if (model_name == 'STGCN') or  (model_name == 'stgcn'):
        # Utilise la distance adjacency matrix 
        config = dict(model_name= model_name,epochs = [3], lr = [1e-4],batch_size = [64],
                        dropout = [0.2],calib_prop = [0.5],
                        enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',
                        
                        num_nodes = 40, n_his=8, n_pred = 1, time_intvl = 5, Kt = 3, stblock_num=2,act_func=['glu','gtu'],Ks = [3,2],
                        graph_conv_type = ['cheb_graph_conv', 'graph_conv'],gso_type = ['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'],
                        enable_bias = 'True',out_dim = 2,adj_type = 'dist',enable_padding = True,

                        threeshold = 0.3,gamma = 0.95,patience = 30,scheduler = None, ray = False
        )



    if model_name== 'LSTM':
        config = dict(model_name= model_name,epochs = [30], lr = [1e-4],batch_size = [64],
                        dropout = [0.2],calib_prop = [0.5], 
                        enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',

                          h_dim =[16,32,64],C_outs = [[16,2],[32,2],[16,16,2]],num_layers = 2,bias = True,
                         bidirectional = [True,False], scheduler = None, ray = False
        )
        
    if model_name == 'GRU':
        config = dict(model_name= model_name,epochs = [30], lr = [1e-4],batch_size = [64],
                        dropout = [0.2],calib_prop = [0.5],
                        enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',

                         h_dim =[16,32,64],C_outs = [[16,2],[32,2],[16,16,2]],num_layers = 2,bias = True,
                         bidirectional = [True,False], scheduler = None, ray = False
        )

    if model_name == 'RNN':
        config = dict(model_name= model_name,epochs = [30], lr = [1e-4],batch_size = [64],
                        dropout = [0.2],calib_prop = [0.5],
                        enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',

                        h_dim =[16,32,64],C_outs = [[16,2],[32,2],[16,16,2]],num_layers = 2,bias = True,
                         bidirectional = [True,False], scheduler = None , ray = False
        )


    for key in other_params.keys():
        config[key] = other_params[key]
        
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['optimizer'] = ['sgd','adam','adamw']
    config['weight_decay'] = 0.0005
    config['momentum'] = 0.99
    config['train_prop'] = 0.6
    config['calib_prop'] = 0.5
    config['valid_prop'] = 0.2  
    config['alpha'] = 0.1
    config['loss_function_type'] = 'quantile'
    config['conformity_scores_type'] = 'max_residual'
    config['quantile_method'] =  'weekday_hour' # 'classic'
    config['calendar_class'] = 3
    config['specific_lr'] = [True, False]
    assert config['train_prop']+ config['valid_prop'] < 1.0, f"train_prop + valid_prop = {config['train_prop']+ config['valid_prop']}. No Testing set"
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


def get_config_embed(nb_words_embedding,embedding_dim = 2,position = 'input'):
    '''
    args
    -----
    nb_words_embedding : represent the number of expected class from tuple (weekday,hour,minute) 
    '''
    config_Tembed = dict(nb_words_embedding= nb_words_embedding,embedding_dim = embedding_dim, position=position)
    return(config_Tembed)


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


def display_config(args,args_embedding):
    optimizer = f"Optimizer: {args.optimizer}"
    lr = 'A specific LR by layer is used' if args.specific_lr else 'The same LR is used for each layer'
    calendar_class = f"Calendar class: {args.calendar_class}"
    quantile_method = f"Quantile Method: {args.quantile_method}"

    encoding = f"Encoding dimension: {args_embedding.nb_words_embedding}. Is related to Dictionnary size of the Temporal Embedding Layer"
    embedding_dim = f"Embedding dimension: {args_embedding.embedding_dim}"
    position = f"Position of the Embedding layer: {args_embedding.position}"
    print(f"Model : {args.model_name} \n {optimizer} \n {lr} \n {calendar_class} \n {quantile_method} \n {encoding} \n {embedding_dim} \n {position} ")
