constant_name = ''
constant_modif = {}

modifications = {
    'HPO_subway_in_calendar_embedding_netmob_POIs_STGCN_HuberLossLoss_2025_08_11_23_22_40073_h1': {
            'model_name': 'STGCN',
            'target_data': 'subway_in',
             'dataset_names': ['subway_in', 'calendar_embedding', 'netmob_POIs'],
            'dataset_for_coverage': ['subway_in', 'netmob_POIs'],
            'calendar_types': ['dayofweek', 'timeofday'],
            'embedding_calendar_types': ['dayofweek', 'hour'],


            # --- Contextual NetMob
            'denoising_names': ['netmob_POIs'],
            'denoiser_names': ['exponential'],
            'denoising_modes': ['train', 'valid', 'test'],
            'denoiser_kwargs': {'exponential': {'alpha': 0.8}},
            'contextual_kwargs': {'netmob_POIs': {'need_global_attn': True,
                                                    'stacked_contextual': False,
                                                    'NetMob_selected_apps': ['Google_Maps', 'Web_Weather'],
                                                    'NetMob_transfer_mode': ['DL'],
                                                    'NetMob_selected_tags': ['iris'],
                                                    'NetMob_expanded': '',
                                                    'NetMob_only_epsilon': False,
                                                    'vision_model_name': None,
                                                    'epsilon_clustering': 0.1,
                                                    'agg_iris_target_n': None,
                                                    'use_only_for_common_dates': False,
                                                    'attn_kwargs': {'dim_feedforward': 128,
                                                        'num_heads': 4,
                                                        'dim_model': 64,
                                                        'keep_topk': 30,
                                                        'nb_layers': 3,
                                                        'latent_dim': 64},
                                                },
                                },
            # ----

            # ---- Compilation : 
            'num_workers': 0,
            'persistent_workers': False,
            'pin_memory': False,
            'prefetch_factor': None,
            'drop_last': False,
            'mixed_precision': False,
            'non_blocking': True,
            'torch_compile': False,
            'backend': 'inductor',
            'prefetch_all': False,
            # ----


            # ---- Preprocessing : 
            'data_augmentation': False,
            'minmaxnorm': True,
            'standardize': False,

            # ---- Training Parameters : 
            'H': 6,
            'W': 0,
            'D': 1,
            'step_ahead': 1,
            'horizon_step': 1,
            'out_dim_factor': 1,
            'shuffle': True,
            'train_prop': 0.6,
            'calib_prop': None,
            'valid_prop': 0.2,
            'test_prop': 0.19999999999999996,
            'min_fold_size_proportion': 0.75,
            'K_fold': 2,
            'freq': '15min',
            'train_pourcent': 100,
            'validation_split_method': 'forward_chaining_cv',
            'train_valid_test_split_method': 'similar_length_method',
            'use_target_as_context': False,
            'no_common_dates_between_set': False,
            'unormalize_loss': True,
            # ----

            # ---- STGCN Architecture :
            'temporal_graph_transformer_encoder': False,
            'Kt': 2,
            'stblock_num': 4,
            'Ks': 2,
            'graph_conv_type': 'graph_conv',
            'gso_type': 'sym_renorm_adj',
            'enable_bias': True,
            'adj_type': 'corr',
            'enable_padding': True,
            'threshold': 0.3,
            'act_func': 'glu',
            'temporal_h_dim': 64,
            'spatial_h_dim': 256,
            'output_h_dim': 64,

            # ---- Optimization Parameters :
            'weight_decay': 0.0006214716516792,
            'batch_size': 128,
            'lr': 0.0002481832349353,
            'dropout': 0.145169206052754,
            'epochs': 1000,
            'scheduler': None,
            'torch_scheduler': None,
            'optimizer': 'adamw',
            'loss_function_type': 'HuberLoss',
            # ----



            # ---- Calendar embedding : 
            'TE_embedding_dim': 64,
            'TE_out_h_dim': 64,
            'TE_concatenation_late': True,
            'TE_concatenation_early': False,
            'TE_multi_embedding': True,
            'TE_specific_lr': False,
            'TE_concatenation_late': True,
            'TE_concatenation_early': False,
            'TE_variable_selection_model_name': 'MLP',
            'TE_embedding_dim_calendar_units': [3, 5],
            # ---- 


            # ---- Evaluation : : 
            'evaluate_complete_ds': True,
            'metrics': ['rmse', 'mse', 'mae', 'mape', 'mase'],
            'track_pi': False,
            'track_grad_norm': True,
            # ---- 

            # ---- Others : 
            'need_global_attn': False,
            'learnable_adj_matrix': False,
            'single_station': False,
            'quick_vision': False,
            'set_spatial_units': None,
            'hp_tuning_on_first_fold': True,
            'keep_best_weights': False,
            'ray': False,
            'alpha': None,
            # ---- 
             } ,

'Best_without_netmob_h1': {
                        'target_data': 'subway_in',
                        'dataset_names': ['subway_in', 'calendar_embedding'],
                        'dataset_for_coverage': ['subway_in','netmob_POIs'],
                        'embedding_calendar_types': ['dayofweek', 'hour'],
                        'loss_function_type':'HuberLoss',
                        'Kt': 2,
                        'stblock_num': 4,
                        'Ks': 2,
                        'graph_conv_type': 'graph_conv',
                        'gso_type': 'sym_renorm_adj',
                        'enable_bias': True,
                        'adj_type': 'corr',
                        'enable_padding': True,
                        'threshold': 0.3,
                        'act_func': 'glu',
                        'temporal_h_dim': 64,
                        'spatial_h_dim': 256,
                        'output_h_dim': 64,
                        'weight_decay': 0.0014517707449388,
                        'batch_size': 128,
                        'lr': 0.00071,
                        'dropout': 0.145169206052754,
                        'epochs': 500,
                        'standardize': False,
                        'minmaxnorm': True,

                        'step_ahead': 1,
                        'horizon_step': 1,
                        

                        'TE_embedding_dim': 64,
                        'TE_out_h_dim': 64,
                        'TE_concatenation_late': True,
                        'TE_concatenation_early':False,

                        'use_target_as_context': False,

                        'optimizer': 'adamw',
                        'batch_size': 128,
                        'freq': '15min',
                        'H':6,
                        'D':1,
                        'W':0,

                        'ray': False,
                        'contextual_kwargs' : {},
                        'denoising_names':[],
                        }, 

  'HPO_subway_in_calendar_embedding_netmob_POIs_STGCN_HuberLossLoss_2025_08_11_23_22_40073_h2': {
            'model_name': 'STGCN',
            'target_data': 'subway_in',
             'dataset_names': ['subway_in', 'calendar_embedding', 'netmob_POIs'],
            'dataset_for_coverage': ['subway_in', 'netmob_POIs'],
            'calendar_types': ['dayofweek', 'timeofday'],
            'embedding_calendar_types': ['dayofweek', 'hour'],


            # --- Contextual NetMob
            'denoising_names': ['netmob_POIs'],
            'denoiser_names': ['exponential'],
            'denoising_modes': ['train', 'valid', 'test'],
            'denoiser_kwargs': {'exponential': {'alpha': 0.8}},
            'contextual_kwargs': {'netmob_POIs': {'need_global_attn': True,
                                                    'stacked_contextual': False,
                                                    'NetMob_selected_apps': ['Google_Maps', 'Web_Weather'],
                                                    'NetMob_transfer_mode': ['DL'],
                                                    'NetMob_selected_tags': ['iris'],
                                                    'NetMob_expanded': '',
                                                    'NetMob_only_epsilon': False,
                                                    'vision_model_name': None,
                                                    'epsilon_clustering': 0.1,
                                                    'agg_iris_target_n': None,
                                                    'use_only_for_common_dates': False,
                                                    'attn_kwargs': {'dim_feedforward': 128,
                                                        'num_heads': 4,
                                                        'dim_model': 64,
                                                        'keep_topk': 30,
                                                        'nb_layers': 3,
                                                        'latent_dim': 64},
                                                },
                                },
            # ----

            # ---- Compilation : 
            'num_workers': 0,
            'persistent_workers': False,
            'pin_memory': False,
            'prefetch_factor': None,
            'drop_last': False,
            'mixed_precision': False,
            'non_blocking': True,
            'torch_compile': False,
            'backend': 'inductor',
            'prefetch_all': False,
            # ----


            # ---- Preprocessing : 
            'data_augmentation': False,
            'minmaxnorm': True,
            'standardize': False,

            # ---- Training Parameters : 
            'H': 6,
            'W': 0,
            'D': 1,
            'step_ahead': 2,
            'horizon_step': 2,
            'out_dim_factor': 1,
            'shuffle': True,
            'train_prop': 0.6,
            'calib_prop': None,
            'valid_prop': 0.2,
            'test_prop': 0.19999999999999996,
            'min_fold_size_proportion': 0.75,
            'K_fold': 2,
            'freq': '15min',
            'train_pourcent': 100,
            'validation_split_method': 'forward_chaining_cv',
            'train_valid_test_split_method': 'similar_length_method',
            'use_target_as_context': False,
            'no_common_dates_between_set': False,
            'unormalize_loss': True,
            # ----

            # ---- STGCN Architecture :
            'temporal_graph_transformer_encoder': False,
            'Kt': 2,
            'stblock_num': 4,
            'Ks': 2,
            'graph_conv_type': 'graph_conv',
            'gso_type': 'sym_renorm_adj',
            'enable_bias': True,
            'adj_type': 'corr',
            'enable_padding': True,
            'threshold': 0.3,
            'act_func': 'glu',
            'temporal_h_dim': 64,
            'spatial_h_dim': 256,
            'output_h_dim': 64,

            # ---- Optimization Parameters :
            'weight_decay': 0.0006214716516792,
            'batch_size': 128,
            'lr': 0.0002481832349353,
            'dropout': 0.145169206052754,
            'epochs': 1000,
            'scheduler': None,
            'torch_scheduler': None,
            'optimizer': 'adamw',
            'loss_function_type': 'HuberLoss',
            # ----



            # ---- Calendar embedding : 
            'TE_embedding_dim': 64,
            'TE_out_h_dim': 64,
            'TE_concatenation_late': True,
            'TE_concatenation_early': False,
            'TE_multi_embedding': True,
            'TE_specific_lr': False,
            'TE_concatenation_late': True,
            'TE_concatenation_early': False,
            'TE_variable_selection_model_name': 'MLP',
            'TE_embedding_dim_calendar_units': [3, 5],
            # ---- 


            # ---- Evaluation : : 
            'evaluate_complete_ds': True,
            'metrics': ['rmse', 'mse', 'mae', 'mape', 'mase'],
            'track_pi': False,
            'track_grad_norm': True,
            # ---- 

            # ---- Others : 
            'need_global_attn': False,
            'learnable_adj_matrix': False,
            'single_station': False,
            'quick_vision': False,
            'set_spatial_units': None,
            'hp_tuning_on_first_fold': True,
            'keep_best_weights': False,
            'ray': False,
            'alpha': None,
            # ---- 
             } ,

'Best_without_netmob_h2': {
                        'target_data': 'subway_in',
                        'dataset_names': ['subway_in', 'calendar_embedding'],
                        'dataset_for_coverage': ['subway_in','netmob_POIs'],
                        'embedding_calendar_types': ['dayofweek', 'hour'],
                        'loss_function_type':'HuberLoss',
                        'Kt': 2,
                        'stblock_num': 4,
                        'Ks': 2,
                        'graph_conv_type': 'graph_conv',
                        'gso_type': 'sym_renorm_adj',
                        'enable_bias': True,
                        'adj_type': 'corr',
                        'enable_padding': True,
                        'threshold': 0.3,
                        'act_func': 'glu',
                        'temporal_h_dim': 64,
                        'spatial_h_dim': 256,
                        'output_h_dim': 64,
                        'weight_decay': 0.0014517707449388,
                        'batch_size': 128,
                        'lr': 0.00071,
                        'dropout': 0.145169206052754,
                        'epochs': 500,
                        'standardize': False,
                        'minmaxnorm': True,

                        'step_ahead': 2,
                        'horizon_step': 2,
                        

                        'TE_embedding_dim': 64,
                        'TE_out_h_dim': 64,
                        'TE_concatenation_late': True,
                        'TE_concatenation_early':False,

                        'use_target_as_context': False,

                        'optimizer': 'adamw',
                        'batch_size': 128,
                        'freq': '15min',
                        'H':6,
                        'D':1,
                        'W':0,

                        'ray': False,
                        'contextual_kwargs' : {},
                        'denoising_names':[],
                        }, 


  'HPO_subway_in_calendar_embedding_netmob_POIs_STGCN_HuberLossLoss_2025_08_11_23_22_40073_h3': {
            'model_name': 'STGCN',
            'target_data': 'subway_in',
             'dataset_names': ['subway_in', 'calendar_embedding', 'netmob_POIs'],
            'dataset_for_coverage': ['subway_in', 'netmob_POIs'],
            'calendar_types': ['dayofweek', 'timeofday'],
            'embedding_calendar_types': ['dayofweek', 'hour'],


            # --- Contextual NetMob
            'denoising_names': ['netmob_POIs'],
            'denoiser_names': ['exponential'],
            'denoising_modes': ['train', 'valid', 'test'],
            'denoiser_kwargs': {'exponential': {'alpha': 0.8}},
            'contextual_kwargs': {'netmob_POIs': {'need_global_attn': True,
                                                    'stacked_contextual': False,
                                                    'NetMob_selected_apps': ['Google_Maps', 'Web_Weather'],
                                                    'NetMob_transfer_mode': ['DL'],
                                                    'NetMob_selected_tags': ['iris'],
                                                    'NetMob_expanded': '',
                                                    'NetMob_only_epsilon': False,
                                                    'vision_model_name': None,
                                                    'epsilon_clustering': 0.1,
                                                    'agg_iris_target_n': None,
                                                    'use_only_for_common_dates': False,
                                                    'attn_kwargs': {'dim_feedforward': 128,
                                                        'num_heads': 4,
                                                        'dim_model': 64,
                                                        'keep_topk': 30,
                                                        'nb_layers': 3,
                                                        'latent_dim': 64},
                                                },
                                },
            # ----

            # ---- Compilation : 
            'num_workers': 0,
            'persistent_workers': False,
            'pin_memory': False,
            'prefetch_factor': None,
            'drop_last': False,
            'mixed_precision': False,
            'non_blocking': True,
            'torch_compile': False,
            'backend': 'inductor',
            'prefetch_all': False,
            # ----


            # ---- Preprocessing : 
            'data_augmentation': False,
            'minmaxnorm': True,
            'standardize': False,

            # ---- Training Parameters : 
            'H': 6,
            'W': 0,
            'D': 1,
            'step_ahead': 3,
            'horizon_step': 3,
            'out_dim_factor': 1,
            'shuffle': True,
            'train_prop': 0.6,
            'calib_prop': None,
            'valid_prop': 0.2,
            'test_prop': 0.19999999999999996,
            'min_fold_size_proportion': 0.75,
            'K_fold': 2,
            'freq': '15min',
            'train_pourcent': 100,
            'validation_split_method': 'forward_chaining_cv',
            'train_valid_test_split_method': 'similar_length_method',
            'use_target_as_context': False,
            'no_common_dates_between_set': False,
            'unormalize_loss': True,
            # ----

            # ---- STGCN Architecture :
            'temporal_graph_transformer_encoder': False,
            'Kt': 2,
            'stblock_num': 4,
            'Ks': 2,
            'graph_conv_type': 'graph_conv',
            'gso_type': 'sym_renorm_adj',
            'enable_bias': True,
            'adj_type': 'corr',
            'enable_padding': True,
            'threshold': 0.3,
            'act_func': 'glu',
            'temporal_h_dim': 64,
            'spatial_h_dim': 256,
            'output_h_dim': 64,

            # ---- Optimization Parameters :
            'weight_decay': 0.0006214716516792,
            'batch_size': 128,
            'lr': 0.0002481832349353,
            'dropout': 0.145169206052754,
            'epochs': 1000,
            'scheduler': None,
            'torch_scheduler': None,
            'optimizer': 'adamw',
            'loss_function_type': 'HuberLoss',
            # ----



            # ---- Calendar embedding : 
            'TE_embedding_dim': 64,
            'TE_out_h_dim': 64,
            'TE_concatenation_late': True,
            'TE_concatenation_early': False,
            'TE_multi_embedding': True,
            'TE_specific_lr': False,
            'TE_concatenation_late': True,
            'TE_concatenation_early': False,
            'TE_variable_selection_model_name': 'MLP',
            'TE_embedding_dim_calendar_units': [3, 5],
            # ---- 


            # ---- Evaluation : : 
            'evaluate_complete_ds': True,
            'metrics': ['rmse', 'mse', 'mae', 'mape', 'mase'],
            'track_pi': False,
            'track_grad_norm': True,
            # ---- 

            # ---- Others : 
            'need_global_attn': False,
            'learnable_adj_matrix': False,
            'single_station': False,
            'quick_vision': False,
            'set_spatial_units': None,
            'hp_tuning_on_first_fold': True,
            'keep_best_weights': False,
            'ray': False,
            'alpha': None,
            # ---- 
             } ,

'Best_without_netmob_h3': {
                        'target_data': 'subway_in',
                        'dataset_names': ['subway_in', 'calendar_embedding'],
                        'dataset_for_coverage': ['subway_in','netmob_POIs'],
                        'embedding_calendar_types': ['dayofweek', 'hour'],
                        'loss_function_type':'HuberLoss',
                        'Kt': 2,
                        'stblock_num': 4,
                        'Ks': 2,
                        'graph_conv_type': 'graph_conv',
                        'gso_type': 'sym_renorm_adj',
                        'enable_bias': True,
                        'adj_type': 'corr',
                        'enable_padding': True,
                        'threshold': 0.3,
                        'act_func': 'glu',
                        'temporal_h_dim': 64,
                        'spatial_h_dim': 256,
                        'output_h_dim': 64,
                        'weight_decay': 0.0014517707449388,
                        'batch_size': 128,
                        'lr': 0.00071,
                        'dropout': 0.145169206052754,
                        'epochs': 500,
                        'standardize': False,
                        'minmaxnorm': True,

                        'step_ahead': 3,
                        'horizon_step': 3,
                        

                        'TE_embedding_dim': 64,
                        'TE_out_h_dim': 64,
                        'TE_concatenation_late': True,
                        'TE_concatenation_early':False,

                        'use_target_as_context': False,

                        'optimizer': 'adamw',
                        'batch_size': 128,
                        'freq': '15min',
                        'H':6,
                        'D':1,
                        'W':0,

                        'ray': False,
                        'contextual_kwargs' : {},
                        'denoising_names':[],
                        },   

  'HPO_subway_in_calendar_embedding_netmob_POIs_STGCN_HuberLossLoss_2025_08_11_23_22_40073_h4': {
            'model_name': 'STGCN',
            'target_data': 'subway_in',
             'dataset_names': ['subway_in', 'calendar_embedding', 'netmob_POIs'],
            'dataset_for_coverage': ['subway_in', 'netmob_POIs'],
            'calendar_types': ['dayofweek', 'timeofday'],
            'embedding_calendar_types': ['dayofweek', 'hour'],


            # --- Contextual NetMob
            'denoising_names': ['netmob_POIs'],
            'denoiser_names': ['exponential'],
            'denoising_modes': ['train', 'valid', 'test'],
            'denoiser_kwargs': {'exponential': {'alpha': 0.8}},
            'contextual_kwargs': {'netmob_POIs': {'need_global_attn': True,
                                                    'stacked_contextual': False,
                                                    'NetMob_selected_apps': ['Google_Maps', 'Web_Weather'],
                                                    'NetMob_transfer_mode': ['DL'],
                                                    'NetMob_selected_tags': ['iris'],
                                                    'NetMob_expanded': '',
                                                    'NetMob_only_epsilon': False,
                                                    'vision_model_name': None,
                                                    'epsilon_clustering': 0.1,
                                                    'agg_iris_target_n': None,
                                                    'use_only_for_common_dates': False,
                                                    'attn_kwargs': {'dim_feedforward': 128,
                                                        'num_heads': 4,
                                                        'dim_model': 64,
                                                        'keep_topk': 30,
                                                        'nb_layers': 3,
                                                        'latent_dim': 64},
                                                },
                                },
            # ----

            # ---- Compilation : 
            'num_workers': 0,
            'persistent_workers': False,
            'pin_memory': False,
            'prefetch_factor': None,
            'drop_last': False,
            'mixed_precision': False,
            'non_blocking': True,
            'torch_compile': False,
            'backend': 'inductor',
            'prefetch_all': False,
            # ----


            # ---- Preprocessing : 
            'data_augmentation': False,
            'minmaxnorm': True,
            'standardize': False,

            # ---- Training Parameters : 
            'H': 6,
            'W': 0,
            'D': 1,
            'step_ahead': 4,
            'horizon_step': 4,
            'out_dim_factor': 1,
            'shuffle': True,
            'train_prop': 0.6,
            'calib_prop': None,
            'valid_prop': 0.2,
            'test_prop': 0.19999999999999996,
            'min_fold_size_proportion': 0.75,
            'K_fold': 2,
            'freq': '15min',
            'train_pourcent': 100,
            'validation_split_method': 'forward_chaining_cv',
            'train_valid_test_split_method': 'similar_length_method',
            'use_target_as_context': False,
            'no_common_dates_between_set': False,
            'unormalize_loss': True,
            # ----

            # ---- STGCN Architecture :
            'temporal_graph_transformer_encoder': False,
            'Kt': 2,
            'stblock_num': 4,
            'Ks': 2,
            'graph_conv_type': 'graph_conv',
            'gso_type': 'sym_renorm_adj',
            'enable_bias': True,
            'adj_type': 'corr',
            'enable_padding': True,
            'threshold': 0.3,
            'act_func': 'glu',
            'temporal_h_dim': 64,
            'spatial_h_dim': 256,
            'output_h_dim': 64,

            # ---- Optimization Parameters :
            'weight_decay': 0.0006214716516792,
            'batch_size': 128,
            'lr': 0.0002481832349353,
            'dropout': 0.145169206052754,
            'epochs': 1000,
            'scheduler': None,
            'torch_scheduler': None,
            'optimizer': 'adamw',
            'loss_function_type': 'HuberLoss',
            # ----



            # ---- Calendar embedding : 
            'TE_embedding_dim': 64,
            'TE_out_h_dim': 64,
            'TE_concatenation_late': True,
            'TE_concatenation_early': False,
            'TE_multi_embedding': True,
            'TE_specific_lr': False,
            'TE_concatenation_late': True,
            'TE_concatenation_early': False,
            'TE_variable_selection_model_name': 'MLP',
            'TE_embedding_dim_calendar_units': [3, 5],
            # ---- 


            # ---- Evaluation : : 
            'evaluate_complete_ds': True,
            'metrics': ['rmse', 'mse', 'mae', 'mape', 'mase'],
            'track_pi': False,
            'track_grad_norm': True,
            # ---- 

            # ---- Others : 
            'need_global_attn': False,
            'learnable_adj_matrix': False,
            'single_station': False,
            'quick_vision': False,
            'set_spatial_units': None,
            'hp_tuning_on_first_fold': True,
            'keep_best_weights': False,
            'ray': False,
            'alpha': None,
            # ---- 
             } ,

'Best_without_netmob_h4': {
                        'target_data': 'subway_in',
                        'dataset_names': ['subway_in', 'calendar_embedding'],
                        'dataset_for_coverage': ['subway_in','netmob_POIs'],
                        'embedding_calendar_types': ['dayofweek', 'hour'],
                        'loss_function_type':'HuberLoss',
                        'Kt': 2,
                        'stblock_num': 4,
                        'Ks': 2,
                        'graph_conv_type': 'graph_conv',
                        'gso_type': 'sym_renorm_adj',
                        'enable_bias': True,
                        'adj_type': 'corr',
                        'enable_padding': True,
                        'threshold': 0.3,
                        'act_func': 'glu',
                        'temporal_h_dim': 64,
                        'spatial_h_dim': 256,
                        'output_h_dim': 64,
                        'weight_decay': 0.0014517707449388,
                        'batch_size': 128,
                        'lr': 0.00071,
                        'dropout': 0.145169206052754,
                        'epochs': 500,
                        'standardize': False,
                        'minmaxnorm': True,

                        'step_ahead': 4,
                        'horizon_step': 4,
                        

                        'TE_embedding_dim': 64,
                        'TE_out_h_dim': 64,
                        'TE_concatenation_late': True,
                        'TE_concatenation_early':False,

                        'use_target_as_context': False,

                        'optimizer': 'adamw',
                        'batch_size': 128,
                        'freq': '15min',
                        'H':6,
                        'D':1,
                        'W':0,

                        'ray': False,
                        'contextual_kwargs' : {},
                        'denoising_names':[],
                        },      

}



if len(constant_modif) > 0:
    modifications_bis = modifications.copy()
    modifications = {}
    for key, value in modifications_bis.items():
        modif_i = constant_modif.copy()
        modif_i.update(value)
        modifications[f"{constant_name}_{key}"] = modif_i