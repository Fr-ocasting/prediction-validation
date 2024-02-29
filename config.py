import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config(model_name):

    if model_name== 'CNN':
        config = dict(model_name= model_name,epochs = [300], lr = [1e-4],batch_size = [64]
                    ,dropout = [0.2],calib_prop = [0.3,0.5], alpha = [0.05,0.1],device = device,

                    c_in = 1, C_outs = [[16,16,2]],H_dims = [[16,16,16]],out_dim = 2
                    ) 

    if model_name== 'MTGNN':
            config = dict(model_name= model_name,epochs = [30], lr = [1e-4],batch_size = [64],
                        dropout = [0.2],calib_prop = [0.5], alpha = [0.1],device = device,

                    gcn_true = False, buildA_true = False, gcn_depth = 2,propalpha=0.05,predefined_A=None,# inutile ici car pas de Graph Convolution
                    subgraph_size=20,node_dim=40,tanhalpha=3,static_feat=None,  # inutile aussi, c'est pour la construction de matrice d'adjacence

                    num_nodes = 40,dilation_exponential=1,
                    
                    c_in = 1,conv_channels=32, residual_channels=32, 
                    skip_channels=64, end_channels=128,out_dim=2,layers=3,layer_norm_affline=True
                    )
            
    if model_name== 'LSTM':
        config = dict(model_name= model_name,epochs = [30], lr = [1e-4],batch_size = [64],
                        dropout = [0.2],calib_prop = [0.5], alpha = [0.1],device = device,

                         c_in = 1, h_dim =[16,32,64],C_outs = [[16,2],[32,2],[16,16,2]],num_layers = 2,bias = True,
                         bidirectional = [True,False]
        )
        
    if model_name == 'GRU':
        config = dict(model_name= model_name,epochs = [30], lr = [1e-4],batch_size = [64],
                        dropout = [0.2],calib_prop = [0.5], alpha = [0.1],device = device,

                         c_in = 1, h_dim =[16,32,64],C_outs = [[16,2],[32,2],[16,16,2]],num_layers = 2,bias = True,
                         bidirectional = [True,False]
        )

    if model_name == 'RNN':
        config = dict(model_name= model_name,epochs = [30], lr = [1e-4],batch_size = [64],
                        dropout = [0.2],calib_prop = [0.5], alpha = [0.1],device = device,

                         c_in = 1, h_dim =[16,32,64],C_outs = [[16,2],[32,2],[16,16,2]],num_layers = 2,bias = True,
                         bidirectional = [True,False],
        )
    return(config)