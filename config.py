import torch 
import argparse
import random


def get_config(model_name,learn_graph_structure = None):

    if model_name== 'CNN':
        config = dict(model_name= model_name,epochs = [300], lr = [1e-4],batch_size = [64]
                    ,dropout = [0.2],calib_prop = [0.3,0.5], alpha = [0.05,0.1],
                    enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',

                    c_in = 1, C_outs = [[16,16,2]],H_dims = [[16,16,16]],out_dim = 2
                    ) 

    if model_name== 'MTGNN':
        config = dict(model_name= model_name,epochs = [30], lr = [1e-4],batch_size = [64],
                        dropout = [0.2],calib_prop = [0.5], alpha = [0.1],
                        enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',

                    gcn_true = False, buildA_true = False, gcn_depth = 2,propalpha=0.05,predefined_A=None,# inutile ici car pas de Graph Convolution
                    subgraph_size=20,node_dim=40,tanhalpha=3,static_feat=None,  # inutile aussi, c'est pour la construction de matrice d'adjacence

                    num_nodes = 40,dilation_exponential=1,
                    
                    c_in = 1,conv_channels=32, residual_channels=32, 
                    skip_channels=64, end_channels=128,out_dim=2,layers=3,layer_norm_affline=True
                    )
        if learn_graph_structure:
            config['gcn_true'],config['buildA_true'] = True,True   

    if model_name == 'STGCN':
        # Utilise la distance adjacency matrix 
        config = dict(model_name= model_name,epochs = [30], lr = [1e-4],batch_size = [64],
                        dropout = [0.2],calib_prop = [0.5], alpha = [0.1]
                        ,enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',
                        
                        num_nodes = 40, n_his=8, n_pred = 1, time_intvl = 5, Kt = 3, stblock_num=2,act_func=['glu','gtu'],Ks = [3,2],
                        graph_conv_type = ['cheb_graph_conv', 'graph_conv'],gso_type = ['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'],
                        enable_bias = 'True',out_dim = 2,adj_type = 'dist',enable_padding = True,

                        threeshold = 0.3,gamma = 0.95,patience = 30
        )



    if model_name== 'LSTM':
        config = dict(model_name= model_name,epochs = [30], lr = [1e-4],batch_size = [64],
                        dropout = [0.2],calib_prop = [0.5], alpha = [0.1],
                        enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',

                          h_dim =[16,32,64],C_outs = [[16,2],[32,2],[16,16,2]],num_layers = 2,bias = True,
                         bidirectional = [True,False]
        )
        
    if model_name == 'GRU':
        config = dict(model_name= model_name,epochs = [30], lr = [1e-4],batch_size = [64],
                        dropout = [0.2],calib_prop = [0.5], alpha = [0.1],
                        enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',

                         h_dim =[16,32,64],C_outs = [[16,2],[32,2],[16,16,2]],num_layers = 2,bias = True,
                         bidirectional = [True,False]
        )

    if model_name == 'RNN':
        config = dict(model_name= model_name,epochs = [30], lr = [1e-4],batch_size = [64],
                        dropout = [0.2],calib_prop = [0.5], alpha = [0.1],
                        enable_cuda = torch.cuda.is_available(), seed = 42, dataset = 'subway_15_min',

                        h_dim =[16,32,64],C_outs = [[16,2],[32,2],[16,16,2]],num_layers = 2,bias = True,
                         bidirectional = [True,False],
        )
        
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['optimizer'] = ['sgd','adam','adamw']
    config['weight_decay'] = 0.0005
    config['momentum'] = 0.
    return(config)


def get_parameters(config):
    parser = argparse.ArgumentParser(description=config['model_name'])

    for key in config.keys():
        if type(config[key]) != list:
            default = config[key]
        else:
            ind = random.randint(0,len(config[key])-1)
            default = config[key][ind]
        parser.add_argument(f'--{key}', type=type(default), default=default)

    args = parser.parse_args(args=[])
    return(args)
