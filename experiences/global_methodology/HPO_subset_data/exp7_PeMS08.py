import os 
import sys
import torch 
import torch._dynamo as dynamo; dynamo.graph_break()
torch._dynamo.config.verbose=True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..','..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from examples.train_model_on_k_fold_validation import train_model_on_k_fold_validation

if __name__ == '__main__':
    model_name = 'STAEformer'
    target_data = 'PeMS08_flow'
    epochs = 100 # 200
    dataset_for_coverage = [target_data] 
    dataset_names = [target_data,'calendar']
    modification = {'target_data' :target_data,
                    'ray':True,

                    # Expanding Train & Graph Subset: 
                    'expanding_train': None, # None, #0.5,
                    'graph_subset': None,  #0.5, # None,
                    'batch_size': 128, # 16
                        # ----

                    'grace_period':10,
                    'HP_max_epochs':epochs, #1000, #300,
                    'epochs':epochs,
                    'K_fold': 1,

                    'evaluate_complete_ds' : True,
                    'calendar_types':['dayofweek', 'timeofday'],
                    'dataset_for_coverage': ['PeMS08'],
                    'use_target_as_context': False,
                    'data_augmentation': False,
                    'step_ahead': 12,
                    'station' : [],
                    'freq': '5min',
                    'H':12,
                    'D':0,
                    'W':0,

                    'input_embedding_dim': 24,
                    'tod_embedding_dim': 24,
                    'dow_embedding_dim': 24,
                    'spatial_embedding_dim':0,
                    'adaptive_embedding_dim': 80,

                    'node_dim': 64,
                    'feed_forward_dim': 256,
                    'num_heads': 4,
                    'num_layers': 3,

                    'use_mixed_proj': True,


                    'optimizer': 'adam',
                    'lr': 0.001, # 0.001
                    'weight_decay': 0.0015,
                    'torch_scheduler_type': 'MultiStepLR', #'MultiStepLR', 'warmup'
                    'loss_function_type':'HuberLoss',

                    # if torch_scheduler_type == 'MultiStepLR' :
                    'torch_scheduler_milestone': [25, 45, 65],
                    'torch_scheduler_gamma':0.1,

                    # # if torch_scheduler_type == 'warmup' :
                    # 'torch_scheduler_gamma':0.99,
                    # 'torch_scheduler_lr_start_factor' : 0.1,
                    # 'torch_scheduler_milestone' : 5,
                    

                    'train_prop': 0.6,
                    'valid_prop': 0.2,
                    'test_prop': 0.2,
                    'dropout':0.1,
                    'standardize': True,
                    'minmaxnorm': False,
                    'metrics':['masked_mae','masked_rmse','masked_mape','masked_mse','mae','rmse','mape','mse','mase'],
                    'unormalize_loss' : True,

                    'num_workers' : 0, #4, # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
                    'persistent_workers' : False ,# True 
                    'pin_memory' : False ,# True 
                    'prefetch_factor' : None, # 4, # None, 2,3,4,5 ... 
                    'drop_last' : False,  # True
                    'mixed_precision' : False, # True # False
                    'torch_compile' : False,# 'compile', # 'compile' # 'jit_script' #'trace'
                        }



    if True:
        if True:
            from examples.HP_parameter_choice import hyperparameter_tuning
            from examples.benchmark import local_get_args

            def HP_and_valid_one_config(args,epochs_validation,num_samples):
                # HP Tuning on the first fold
                analysis,trial_id = hyperparameter_tuning(args,num_samples)

                # K-fold validation with best config: 
                modification = {'epochs':epochs_validation,
                                'expanding_train': None,
                                'graph_subset': None,
                                'batch_size' : 16,
                                }
                train_model_on_k_fold_validation(trial_id,load_config=True,save_folder='K_fold_validation/training_with_HP_tuning',modification=modification)
                return trial_id

            
            args = local_get_args(model_name,
                            args_init = None,
                            dataset_names=dataset_names,
                            dataset_for_coverage=dataset_for_coverage,
                            modification = modification
                        )

            epochs_validation = epochs #1000
            num_samples = 400 # 100 # 300 # 200
            HP_and_valid_one_config(args,epochs_validation,num_samples)

        # If HPO worked by need to compute again the 'train_valid_k_models':
        # Or if we need to compute with B = 128 
        else:
            #  -------- -------- -------- -------- -------- -------- -------- --------
            # HPO - pas bon  --------    Bcp d'hyperparamètre, pas assez de samples --------
            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_10_31_01_50_36105' 
            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_07_23_56_52940'  
            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_08_06_36_95634'
            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_10_06_35_64908'

            # HPO - Warmup
            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_13_11_11_61374' # expanding_train = 0.2, graph_subset = 0.5
            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_14_22_20_94406' # expanding_train = 0.5, graph_subset = 0.5
            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_16_03_07_14720' # expanding_train = 0.5, graph_subset = None
            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_17_13_33_92297' # expanding_train = None, graph_subset = None

            # HPO - Multi-Step LR 
            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_17_23_12_56159'   # expanding_train = 0.2, graph_subset = 0.5
            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_18_23_01_21887'   # expanding_train = 0.2, graph_subset = None
            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_19_20_19_87308'    # expanding_train = 0.5, graph_subset = None
            # HPO - pas bon  --------            # HPO - pas bon 
            #  -------- -------- -------- -------- -------- -------- -------- --------

            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_20_15_37_86605'   # expanding_train = 0.2, graph_subset = 0.5
            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_21_01_01_63861'   # expanding_train = 0.2, graph_subset = None
            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_21_20_02_66614'   # expanding_train = 0.5, graph_subset = 0.5
            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_22_11_40_11090'   # expanding_train = 0.5, graph_subset = None


            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_26_17_45_76159'     # expanding_train = 0.5, graph_subset = None,  samples =100 
            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_28_21_15_41764' # expanding_train = None, graph_subset = None, samples =100 


            # ------- ONLY WARMUP 
            # trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_29_07_04_62976' # expanding_train = None, graph_subset = None, samples =100 


            # --- Parameter: 
            # optimizer: adam 
            # standardize : True
            # minmaxnorm : False
            # 'torch_scheduler_type': 'MultiStepLR',
            # 'torch_scheduler_milestone': [25, 45, 65],
            # 'torch_scheduler_gamma':0.1,
            # 'expanding_train' : 0.3  (!)
            # 'graph_subset': O.5  (!)
            # HPO: 11h 
            # Training full model with B = 16 : 69min (Troughput: 299.35 sequences / s )
            # Training full model with B = 128 : 42 min (Troughput: 530.14 sequences / s )
            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2026_02_08_01_58_94101' 


            # HPO: 18hr 43min
            # Training full model with B = 16 : 69min (Troughput: 299.35 sequences / s )
            # Training full model with B = 128 : 22min (Troughput: 961.86 sequences / s )
            trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2026_02_08_21_59_97586:'
            



            for trial_id in ['PeMS08_flow_calendar_STAEformer_HuberLossLoss_2026_02_08_21_59_97586'
                             ]:
                modification = {'epochs': epochs, #1,
                                'expanding_train': None,
                                'graph_subset': None,
                                'batch_size' : 128,
                                'device' : torch.device('cuda:1'),
                                'torch_compile' :'compile'
                                }
                train_model_on_k_fold_validation(trial_id,load_config=True,save_folder='K_fold_validation/training_with_HP_tuning',modification=modification)

    # Results HPO 08 - Février - 2026 :
        # --- Parameter: 
        # optimizer: adam 
        # batch-size : 128
        # standardize : True
        # minmaxnorm : False
        # 'torch_scheduler_type': 'MultiStepLR',
        # 'torch_scheduler_milestone': [25, 45, 65],
        # 'torch_scheduler_gamma':0.1,
        # HPO: 11h
        # 'expanding_train' : 0.3  (!)
        # 'graph_subset': O.5  (!)
        # search space: 
        #   "lr": tune.loguniform(5e-5, 5e-3)
        #   "weight_decay" : tune.loguniform(5e-4, 1e-2)
        #   "input_embedding_dim": tune.choice([16,32,64]),
        #   "tod_embedding_dim": tune.choice([4,12,24]),
        #   "dow_embedding_dim": tune.choice([4,12,24]),
        #   "adaptive_embedding_dim": tune.choice([16,32,64]),
        
        # trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2026_02_08_01_58_94101' 
        # All Steps RMSE = 23.376, MAE = 13.497, MASE = 0.849, MAPE = 8.896  (B = 16)
        # All Steps RMSE = 24.037, MAE = 14.846, MASE = 0.934, MAPE = 10.032 (B = 128)

        # ----- 
        # HPO: 18hr 43min
        # 'expanding_train' : 0.3  (!)
        # 'graph_subset': None  (!)

        # trial_id = 'PeMS08_flow_calendar_STAEformer_HuberLossLoss_2026_02_08_21_59_97586'  
        # All Steps RMSE = 23.528, MAE = 13.449, MASE = 0.846, MAPE = 8.839  (B = 16)
        # All Steps RMSE = 23.870, MAE = 14.713, MASE = 0.926, MAPE = 10.014 (B = 128)


        
    # ===================================================================================================================        
    # ===================================================================================================================
    # Results HPO avec MULTISTEP LR : 
        # 'batch_size': 128, 
        # 'epochs' : 100,
        # 'grace_period': 5,
        # 'num_samples': 200,
        # Testing : 
        #   'batch_size': 16


        # ------------------------------------------------------------------
        # 'expanding_train': 0.2,
        # 'graph_subset': 0.5,
        #  Time: ~4h

        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_20_15_37_86605:   All Steps RMSE = 23.874, MAE = 13.839, MASE = 0.871, MAPE = 8.949
        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_20_15_37_86605:   All Steps RMSE = 23.625, MAE = 13.754, MASE = 0.866, MAPE = 9.173 ( B = 128)

        # ------------------------------------------------------------------
        # 'expanding_train': 0.2,
        # 'graph_subset': None,
        #  Time: ~6h45
        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_21_01_01_63861:   All Steps RMSE = 23.309, MAE = 13.744, MASE = 0.865, MAPE = 9.374
        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_21_01_01_63861:   All Steps RMSE = 23.038, MAE = 13.701, MASE = 0.862, MAPE = 9.225


        # ------------------------------------------------------------------
        # 'expanding_train': 0.5,
        # 'graph_subset': 0.5,
        #  Time: ~ 10h

        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_21_20_02_66614:   All Steps RMSE = 23.322, MAE = 13.700, MASE = 0.862, MAPE = 9.120
        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_21_20_02_66614:   All Steps RMSE = 23.270, MAE = 13.692, MASE = 0.862, MAPE = 9.193


        # ------------------------------------------------------------------
        # 'expanding_train': 0.5,
        # 'graph_subset': None,
        #  Time: ~ 13h

        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_22_11_40_11090:   All Steps RMSE = 23.751, MAE = 13.701, MASE = 0.862, MAPE = 8.984
        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_22_11_40_11090:   All Steps RMSE = 23.264, MAE = 13.698, MASE = 0.862, MAPE = 9.130

        # ------------------------------------------------------------------
        # MOEDIFIED : NO MILESTONE HPO
        # 'expanding_train': 0.5,
        # 'graph_subset': None,
        #  Time: ~ 7 h

        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_26_17_45_76159:   All Steps RMSE = 23.313, MAE = 13.504, MASE = 0.850, MAPE = 9.067
        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_26_17_45_76159:   All Steps RMSE = 23.143, MAE = 13.776, MASE = 0.867, MAPE = 9.266 (B = 128)


        # ------------------------------------------------------------------
        # 'expanding_train': None,
        # 'graph_subset': None,
        #  Time: ~ 11h
        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_28_21_15_41764:   All Steps RMSE = 23.200, MAE = 13.501, MASE = 0.850, MAPE = 8.883
        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_28_21_15_41764:   All Steps RMSE = 24.896, MAE = 15.522, MASE = 0.977, MAPE = 11.027 (B = 128)






        # ------------------------------------------------------------------
        # SPECIAL LR SCHEDULER:  warmup [lr start factor : 0.1 / Milestone = 5]
        # 'expanding_train': None,
        # 'graph_subset': None,
        #  Time: ~ 11h
        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_29_07_04_62976:   All Steps RMSE = 23.247, MAE = 13.739, MASE = 0.865, MAPE = 9.279
        #  PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_29_07_04_62976:   All Steps RMSE = 23.770, MAE = 13.746, MASE = 0.865, MAPE = 9.016 (B = 128)













    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Results PAS BON ----- HPO avec MULTISTEP LR : 
        # 'batch_size': 128, 
        # 'epochs' : 100,
        # 'grace_period': 5,
        # 'num_samples': 300,
        # Testing : 
        #   'batch_size': 16


        # ------------------------------------------------------------------
        # 'expanding_train': 0.2,
        # 'graph_subset': 0.5,
        #  Time: ~5h

        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_17_23_12_56159:   All Steps RMSE = 23.203, MAE = 13.813, MASE = 0.869, MAPE = 9.285
        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_17_23_12_56159:   All Steps RMSE = 23.296, MAE = 13.706, MASE = 0.863, MAPE = 9.078   (B = 128)



        # ------------------------------------------------------------------
        # 'expanding_train': 0.2,
        # 'graph_subset': None,
        #  Time: ~ 11h30
        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_18_23_01_21887:   All Steps RMSE = 23.281, MAE = 13.709, MASE = 0.863, MAPE = 9.211 
        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_18_23_01_21887:   All Steps RMSE = 23.562, MAE = 13.758, MASE = 0.866, MAPE = 9.110   (B = 128)


        # ------------------------------------------------------------------
        # 'expanding_train': 0.5,
        # 'graph_subset': None,
        #  Time: ~ 20h

        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_19_20_19_87308:   All Steps RMSE = 23.958, MAE = 13.769, MASE = 0.866, MAPE = 9.035
        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_19_20_19_87308:   All Steps RMSE = 23.363, MAE = 13.745, MASE = 0.865, MAPE = 9.203






    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Results HPO avec WARMUP : 
        # 'batch_size': 128, 
        # 'epochs' : 100,
        # 'grace_period': 5,
        # 'num_samples': 300,
        # Testing : 
        #   'batch_size': 16


        # ------------------------------------------------------------------
        # 'expanding_train': 0.2,
        # 'graph_subset': 0.5,
        #  Time: ~5h

        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_13_20_36_39246:   All Steps RMSE = 23.349, MAE = 13.776, MASE = 0.867, MAPE = 9.153

        # ------------------------------------------------------------------
        # 'expanding_train': 0.5,
        # 'graph_subset': 0.5,
        # Time: ~8h

        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_14_22_20_94406:   All Steps RMSE = 23.695, MAE = 13.888, MASE = 0.874, MAPE = 9.237
        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_14_22_20_94406:   All Steps RMSE = 23.573, MAE = 13.884, MASE = 0.874, MAPE = 9.289   # (B = 128)


        # ------------------------------------------------------------------
        # 'expanding_train': 0.5,
        # 'graph_subset': None,
        # Time: ~ 11h

        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_16_03_07_14720:   All Steps RMSE = 23.769, MAE = 13.758, MASE = 0.866, MAPE = 8.986
        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_16_03_07_14720:   All Steps RMSE = 23.324, MAE = 13.624, MASE = 0.857, MAPE = 9.051  (B = 128)

        # ------------------------------------------------------------------
        # 'expanding_train': None,
        # 'graph_subset': None,
        # Time: ~ 25 h 

        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_17_13_33_92297:   All Steps RMSE = 23.853, MAE = 14.399, MASE = 0.906, MAPE = 10.075  ( OVERFITTING )
        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_17_13_33_92297:   All Steps RMSE = 23.783, MAE = 13.992, MASE = 0.880, MAPE = 9.415  (B = 128)





    # Results HPO : (PAS BON: SCHEDULER MULTISTEP LR AVEC SEARCH SPACE DE WARMUP)

        # ------------------------------------------------------------------
        # 'expanding_train': 0.2,
        # 'graph_subset': 0.3,
        # 'batch_size': 128, 
        # 'epochs' : 100,
        # 'grace_period': 5,
        # 'num_samples': 300,

        # Time: ~2h

        # Testing : 
        #   'batch_size': 16

        #    All Steps RMSE = 24.002, MAE = 13.855, MASE = 0.872, MAPE = 9.526


        # ------------------------------------------------------------------
        # 'expanding_train': 0.2,
        # 'graph_subset': 0.4,
        # 'batch_size': 128, 
        # 'epochs' : 100,
        # 'grace_period': 5,
        # 'num_samples': 300,

        # Time: 2h28

        # Testing : 
        #   'batch_size': 16

        #   All Steps RMSE = 23.675, MAE = 13.755, MASE = 0.866, MAPE = 9.036


        # ------------------------------------------------------------------
        # 'expanding_train': 0.2,
        # 'graph_subset': 0.8,
        # 'batch_size': 128, 
        # 'epochs' : 100,
        # 'grace_period': 5,
        # 'num_samples': 300,

        # Time: 8h

        # Testing : 
        #   'batch_size': 16

        #   All Steps RMSE = PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_07_23_56_52940:   All Steps RMSE = 23.536, MAE = 14.140, MASE = 0.890, MAPE = 9.641
        #  batch-size = 128: PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_07_23_56_52940:   All Steps RMSE = 23.200, MAE = 13.779, MASE = 0.867, MAPE = 9.244


        # ------------------------------------------------------------------
        # 'expanding_train': 0.2,
        # 'graph_subset': None,
        # 'batch_size': 128, 
        # 'epochs' : 100,
        # 'grace_period': 5,
        # 'num_samples': 300,

        # Time: 6h21

        # Testing : 
        #   'batch_size': 16

        #   All Steps RMSE = PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_08_06_36_95634:   All Steps RMSE = 23.250, MAE = 13.689, MASE = 0.861, MAPE = 9.516
        #   All Steps RMSE = PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_08_06_36_95634:   All Steps RMSE = 23.237, MAE = 13.756, MASE = 0.866, MAPE = 9.245 

        # ------------------------------------------------------------------
        # 'expanding_train': 0.4,
        # 'graph_subset': None,
        # 'batch_size': 128, 
        # 'epochs' : 100,
        # 'grace_period': 5,
        # 'num_samples': 300,

        # Time: 8h35

        # Testing : 
        #   'batch_size': 16
        
        #  PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_09_07_40_59836:   All Steps RMSE = 23.347, MAE = 14.092, MASE = 0.887, MAPE = 9.499        # 
        # 
        # 
        # ------------------------------------------------------------------
        # 'expanding_train': 0.8,
        # 'graph_subset': None,
        # 'batch_size': 128, 
        # 'epochs' : 100,
        # 'grace_period': 5,
        # 'num_samples': 300,

        # Time: 8h35

        # Testing : 
        #   'batch_size': 16
        
        #  PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_10_06_35_64908:   All Steps RMSE = 23.730, MAE = 13.980, MASE = 0.880, MAPE = 9.826
        #  B = 128 : PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_10_06_35_64908:   All Steps RMSE = 23.972, MAE = 14.013, MASE = 0.882, MAPE = 9.342

        # ------------------------------------------------------------------
        # 'expanding_train': None,
        # 'graph_subset': None,
        # 'batch_size': 128, 
        # 'epochs' : 100,
        # 'grace_period': 5,
        # 'num_samples': 300,

        # Time: 20h

        # Testing :   
        #   'batch_size': 16
        
        # PeMS08_flow_calendar_STAEformer_HuberLossLoss_2025_11_13_11_11_61374:   All Steps RMSE = 81.581, MAE = 62.806, MASE = 3.952, MAPE = 104.501

        # ------------------------------------------------------------------
        # 'expanding_train': 0.2,  WARMUP 
        # 'graph_subset': 0.5,
        # 'batch_size': 128, 
        # 'epochs' : 100,
        # 'grace_period': 5,
        # 'num_samples': 300,
        # 

        # Time: 

        # Testing :   
        #   'batch_size': 16
        
        #
    else: 
        from pipeline.K_fold_validation.K_fold_validation import KFoldSplitter
        from pipeline.high_level_DL_method import load_optimizer_and_scheduler
        from pipeline.dl_models.full_model import full_model
        from pipeline.trainer import Trainer
        from pipeline.utils.loger import LOG
        from experiences.loop_train_save_log import loop_train_save_log
        loger = LOG()
        # If True get table performance. Else,  Minimal Test:  
        if True:
            modification['expanding_train'] = None
            modification['graph_subset'] = None
            modification['batch_size'] = 16 # 16
            modification['epochs'] = 45
            modification['model_name'] = model_name
            modification['target_data'] = target_data 
            modification['dataset_for_coverage'] = dataset_for_coverage
            modification['dataset_names'] = dataset_names
            modification['ray'] = False
            modification['K_fold'] = 1
            modification['device'] = torch.device('cuda:0')
            modification['optimizer'] = 'adam'
            modification['torch_compile'] = False # 'compile'

        dic_configs = {'PeMS08_STAEformer_paper_config': modification}
        loop_train_save_log(loger,dic_configs,init_save_folder = 'K_fold_validation/training_wo_HP_tuning/benchmark_HPO_bis')

        # Notes :
        # Avec AdamW - B = 16  : MAE = 13.65 / 13.60 / 13.69

        # Avec Adam  - B = 128 : MAE = 14.272 / 14.193 / 14.146

        # Avec Adam  - B = 16  : MAE =  13.38 / 13.45

