constant_name = ''
constant_modif = {
                        }

SEED = 1


config =  {'target_data': 'subway_in',
            'dataset_names': ['subway_in'],
            'dataset_for_coverage': ['subway_in'],
            'use_target_as_context': False,
            'freq': '15min',
            'H':6,
            'D':1,
            'W':0,

            # Architecture:           
            'adj_type': 'corr', # dist # adj 
            'cl_decay_steps': 1000,
            'use_curriculum_learning': False,
            'input_dim': 1,
            'max_diffusion_step': 2,
            'filter_type': 'dual_random_walk', # 'laplacian' # 'dual_random_walk'
            'num_rnn_layers': 2,
            'rnn_units': 64,
            'threshold': 0.3, 


            # Hyperparameters
            'loss_function_type':'masked_mae',
            'optimizer': 'adam',
            'weight_decay': 0.001,
            'batch_size': 128,
            'lr': 0.001,
            'dropout': 0.0,
            'epochs':500,
            'standardize': False,
            'minmaxnorm': True,
            'unormalize_loss' : True,
              }