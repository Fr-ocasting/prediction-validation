constant_name = ''
constant_modif = {}
SEED = 1


config =  {'target_data': 'subway_in',
            'dataset_names': ['subway_in', 'calendar_embedding'],
            'dataset_for_coverage': ['subway_in'],
            'embedding_calendar_types': ['dayofweek', 'hour'],
            'use_target_as_context': False,
            'freq': '15min',
            'H':6,
            'D':1,
            'W':0,

            # Architecture:           
            'adj_type': 'adj', # 'corr', # dist # adj 
            'cl_decay_steps': 1000,   # if use_curriculum_learning == False, then useless
            'use_curriculum_learning': False,
            'input_dim': 1,
            'max_diffusion_step': 2,
            'filter_type': 'random_walk', # 'laplacian' # 'dual_random_walk'
            'num_rnn_layers': 2,
            'rnn_units': 64,
            'threshold': 0.7, 


            # Hyperparameters
            'loss_function_type':'HuberLoss',
            'optimizer': 'adamw',
            'weight_decay': 0.0015,
            'batch_size': 128,
            'lr': 0.005,
            'dropout': 0.1,  # No Dropout in DCRNN
            'epochs':500,
            'standardize': False,
            'minmaxnorm': True,
            'unormalize_loss' : True,

            # Calendar Embedding 
            'TE_embedding_dim': 64,
            'TE_out_h_dim': 64,
            'TE_concatenation_late': True,
            'TE_concatenation_early':False,
              }