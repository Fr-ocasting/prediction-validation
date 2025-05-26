SEED = 1

config = {'dataset_names' :['PeMS08_flow'],#['PeMS08_flow','calendar'], #['PeMS08_flow','PeMS08_speed','PeMS08_occupancy','calendar'],
          'target_data': 'PeMS08_flow',
          'dataset_for_coverage': ['PeMS08'],
          'model_name': 'STGCN',
          'use_target_as_context': False,
          'data_augmentation': False,

          'step_ahead': 12,
          'freq': '5min',
          'H':12,
          'D':0,
          'W':0,

          'Kt': 3, # 2,3,4 # Kernel Size on the Temporal Dimension
          'stblock_num': 3, # 2,3,4 # Number of STConv-blocks
          'Ks': 3,  # 1,2,3 # Number of iteration within the ChebGraphConv ONLY
          'graph_conv_type': 'cheb_graph_conv', # 'graph_conv','cheb_graph_conv' # Type of graph convolution
          'gso_type': 'sym_renorm_adj', # 'sym_norm_lap','rw_norm_lap','sym_renorm_adj','rw_renorm_adj'  # Type of calcul to compute the gso (Weighted Adjacency Matrix)
          'enable_bias': True, # Enable bias on the output module (FC layers at the output of STGCN)
          'enable_padding': True,  # Enable padding on the Temporal convolution. Suitable for short sequence cause (L' = L-2*(Kt-1)*stblock_num)
          'threshold': 0.3,  # threshold to mask the Weighted Adjacency Matrix based on Gaussian Kernel Distance. < threshold become 0
          'act_func': 'glu', #'glu', 'gtu', 'silu'  # Type of activation function on the output module (FC layers at the output of STGCN)    
          'temporal_h_dim': 64, # Dimension of temporal convolution. Stblocks dims = [temporal_h_dim, spatial_h_dim, temporal_h_dim]
          'spatial_h_dim': 16, # Dimension of spatial graph convolution. Stblocks dims = [temporal_h_dim, spatial_h_dim, temporal_h_dim]
          'output_h_dim': 32, # Dimension of hidden layers in output module

          'calendar_types':['dayofweek', 'timeofday'],

          'adj_type': 'adj', # 'dist' # 'corr' # 'adj'

          'batch_size': 16, # 16, 32, 64
          'epochs':30,
          'optimizer': 'adam',
          'lr': 0.001, # 0.001
          'weight_decay': 0.0015,

          'loss_function_type':'HuberLoss',
          'torch_scheduler': None,

          'train_prop': 0.6,
          'valid_prop': 0.2,
          'test_prop': 0.2,
          'dropout':0.1,
          'standardize': True,
          'minmaxnorm': False,
          'metrics':['masked_mae','masked_rmse','masked_mape','masked_mse','mae','rmse','mape','mse','mase'],

            }
