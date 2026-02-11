# GET PARAMETERS
import os 
import sys
import torch 
import importlib
import torch._dynamo as dynamo; dynamo.graph_break()
torch._dynamo.config.verbose=True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True
# Get Parent folder : 

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from constants.config import local_get_args
from pipeline.utils.loger import LOG
from pipeline.utils.rng import set_seed
from pipeline.high_level_DL_method import model_loading_and_training 
from constants.paths import SAVE_DIRECTORY, FOLDER_PATH
from examples.train_model_on_k_fold_validation import save_model_metrics,get_conditions,keep_track_on_metrics,init_metrics


possible_target_kwargs = {
    'subway_out': {'quantile_filter_outliers': 0.99  },  
     'subway_in': { 'quantile_filter_outliers': 0.99 }, 
     'bike_out': {'agg_iris_target_n':100,
                  'threshold_volume_min': 1,
                  'quantile_filter_outliers': 0.99},
     'bike_in': {'agg_iris_target_n':100,
                  'threshold_volume_min': 1,
                  'quantile_filter_outliers': 0.99},
      }


possible_contextual_kwargs = {

                        'bike_in':{'emb_dim' : 24,
                                         'need_global_attn':False, 
                                        'stacked_contextual': False,
                                        'agg_iris_target_n':100,
                                        'threshold_volume_min': 1,
                                        'attn_kwargs': {},
                                    },
                        'bike_out':{'emb_dim' : 24,
                                         'need_global_attn':False, 
                                        'stacked_contextual': False,
                                         'agg_iris_target_n':100,
                                        'threshold_volume_min': 1,
                                        'attn_kwargs': {},
                                    },

                        # 'weather': {'emb_dim' : 8,
                        #                'need_global_attn':False, 
                        #                 'stacked_contextual': False,
                        #                 'vision_model_name' : None,
                        #                 'use_only_for_common_dates': False,
                        #                 'quantile_filter_outliers': 0.99 ,
                        #                 'use_future_values': 4, # 1hour ahead 
                        #                 'unique_serie': True,   # If true then agregate all (2) weather stations into one unique serie
                        #                 'repeat_spatial': True,  # If true then repeat the weather serie for each node of the target data
                        #                 'attn_kwargs': {},
                        #                                     },  
                        # 'weather': {'need_global_attn':True, 
                        #                 'stacked_contextual': False,
                        #                 'vision_model_name' : None,
                        #                 'use_only_for_common_dates': False,
                        #                 'quantile_filter_outliers': 0.99 ,
                        #                 'use_future_values': 4, # 1hour ahead 
                        #                 'attn_kwargs': {
                        #                     'model_dim': 24, 
                        #                     'latent_dim':  24,# has to be = output_model_dim)
                        #                     'feed_forward_dim':64, 
                        #                     'num_heads':4,
                        #                     'num_layers':3,
                        #                     'mask':False,
                        #                     'keep_temporal_dim': True,
                        #                      'tod_embedding_dim' : 6,
                        #                      'dow_embedding_dim': 6,
                        #                     },
                        #             }, 
                        # 'subway_in': {'emb_dim' : 24,
                        #               'need_global_attn':False, 
                        #                 'stacked_contextual': False,
                        #                 'vision_model_name' : None,
                        #                 'use_only_for_common_dates': False,
                        #                 'quantile_filter_outliers': 0.99 ,
                                        
                        #                 'attn_kwargs': {},
                        #             }, 
                        'subway_in': {'need_global_attn':True, 
                                        'stacked_contextual': False,
                                        'vision_model_name' : None,
                                        'use_only_for_common_dates': False,
                                        'quantile_filter_outliers': 0.99 ,
                                        'attn_kwargs': {
                                            'model_dim': 48, 
                                            'latent_dim':  48,# has to be = output_model_dim)
                                            'feed_forward_dim':128, 
                                            'num_heads':4,
                                            'num_layers':3,
                                            'mask':False,
                                            'keep_temporal_dim': True,
                                             'tod_embedding_dim' : 6,
                                             'dow_embedding_dim': 6,
                                            },
                                    }, 

                        'subway_out': {'need_global_attn':True, 
                                        'stacked_contextual': False,
                                        'vision_model_name' : None,
                                        'use_only_for_common_dates': False,
                                        'quantile_filter_outliers': 0.99 ,
                                        'attn_kwargs': {
                                            'model_dim': 48, 
                                            'latent_dim':  48,# has to be = model_dim)
                                            'feed_forward_dim':128, 
                                            'num_heads':4,
                                            'num_layers':3,
                                            'mask':False,
                                            'keep_temporal_dim': True,
                                             'tod_embedding_dim' : 6,
                                             'dow_embedding_dim': 6,
                                                            },
                                    }, 
                    }


weather_dict_config=  {
'1hAheadRepeatWeatherEmb8': {'emb_dim' : 8,
               'need_global_attn':False, 
                'stacked_contextual': False,
                'vision_model_name' : None,
                'use_only_for_common_dates': False,
                'quantile_filter_outliers': 0.99 ,
                'use_future_values': 4, # 1hour ahead 
                'unique_serie': True,   # If true then agregate all (2) weather stations into one unique serie
                'repeat_spatial': True,  # If true then repeat the weather serie for each node of the target data
                'attn_kwargs': {},
                                    },  
'1hAheadRepeatWeatherEmb12': {'emb_dim' : 12,
               'need_global_attn':False, 
                'stacked_contextual': False,
                'vision_model_name' : None,
                'use_only_for_common_dates': False,
                'quantile_filter_outliers': 0.99 ,
                'use_future_values': 4, # 1hour ahead 
                'unique_serie': True,   # If true then agregate all (2) weather stations into one unique serie
                'repeat_spatial': True,  # If true then repeat the weather serie for each node of the target data
                'attn_kwargs': {},
                                    },  
'1hAheadRepeatWeatherEmb16': {'emb_dim' : 16,
               'need_global_attn':False, 
                'stacked_contextual': False,
                'vision_model_name' : None,
                'use_only_for_common_dates': False,
                'quantile_filter_outliers': 0.99 ,
                'use_future_values': 4, # 1hour ahead 
                'unique_serie': True,   # If true then agregate all (2) weather stations into one unique serie
                'repeat_spatial': True,  # If true then repeat the weather serie for each node of the target data
                'attn_kwargs': {},
                                    },  

'1hAheadWeatherConcatLateAttnH4L3D48FF128':{'need_global_attn':True, 
                'stacked_contextual': False,
                'vision_model_name' : None,
                'use_only_for_common_dates': False,
                'quantile_filter_outliers': 0.99 ,
                'use_future_values': 4, # 1hour ahead 
                'attn_kwargs': {
                    'model_dim': 48, 
                    'latent_dim':  48,# has to be = output_model_dim)
                    'feed_forward_dim':128, 
                    'num_heads':4,
                    'num_layers':3,
                    'mask':False,
                    'keep_temporal_dim': True,
                     'tod_embedding_dim' : 6,
                     'dow_embedding_dim': 6,
                    },
            }, 

'1hAheadWeatherConcatLateAttnH4L3D24FF64':{'need_global_attn':True, 
                'stacked_contextual': False,
                'vision_model_name' : None,
                'use_only_for_common_dates': False,
                'quantile_filter_outliers': 0.99 ,
                'use_future_values': 4, # 1hour ahead 
                'attn_kwargs': {
                    'model_dim': 24, 
                    'latent_dim':  24,# has to be = output_model_dim)
                    'feed_forward_dim':64, 
                    'num_heads':4,
                    'num_layers':3,
                    'mask':False,
                    'keep_temporal_dim': True,
                     'tod_embedding_dim' : 6,
                     'dow_embedding_dim': 6,
                    },
            }, 
'1hAheadWeatherConcatLateAttnH4L3D24FF128':{'need_global_attn':True, 
                'stacked_contextual': False,
                'vision_model_name' : None,
                'use_only_for_common_dates': False,
                'quantile_filter_outliers': 0.99 ,
                'use_future_values': 4, # 1hour ahead 
                'attn_kwargs': {
                    'model_dim': 24, 
                    'latent_dim':  24,# has to be = output_model_dim)
                    'feed_forward_dim':128, 
                    'num_heads':4,
                    'num_layers':3,
                    'mask':False,
                    'keep_temporal_dim': True,
                     'tod_embedding_dim' : 6,
                     'dow_embedding_dim': 6,
                    },
            }, 

'1hAheadWeatherConcatLateAttnH4L1D24FF64':{'need_global_attn':True, 
                'stacked_contextual': False,
                'vision_model_name' : None,
                'use_only_for_common_dates': False,
                'quantile_filter_outliers': 0.99 ,
                'use_future_values': 4, # 1hour ahead 
                'attn_kwargs': {
                    'model_dim': 24, 
                    'latent_dim':  24,# has to be = output_model_dim)
                    'feed_forward_dim':64, 
                    'num_heads':4,
                    'num_layers':3,
                    'mask':False,
                    'keep_temporal_dim': True,
                     'tod_embedding_dim' : 6,
                     'dow_embedding_dim': 6,
                    },
            }, 

'1hAheadWeatherConcatLateAttnH4L1D24FF128':{'need_global_attn':True, 
                'stacked_contextual': False,
                'vision_model_name' : None,
                'use_only_for_common_dates': False,
                'quantile_filter_outliers': 0.99 ,
                'use_future_values': 4, # 1hour ahead 
                'attn_kwargs': {
                    'model_dim': 24, 
                    'latent_dim':  24,# has to be = output_model_dim)
                    'feed_forward_dim':128, 
                    'num_heads':4,
                    'num_layers':1,
                    'mask':False,
                    'keep_temporal_dim': True,
                     'tod_embedding_dim' : 6,
                     'dow_embedding_dim': 6,
                    },
            }, 
}

modifications = {}
# for weather_method in ['1hAheadWeatherConcatLateAttnH4L1D24FF64', '1hAheadWeatherConcatLateAttnH4L1D24FF128',
#                        '1hAheadWeatherConcatLateAttnH4L3D48FF128','1hAheadRepeatWeatherEmb8', '1hAheadRepeatWeatherEmb12', '1hAheadRepeatWeatherEmb16',
#                        '1hAheadWeatherConcatLateAttnH4L3D24FF64','1hAheadWeatherConcatLateAttnH4L3D24FF128',
#                        ]:
for weather_method in ['1hAheadRepeatWeatherEmb16']:
    for target_data in ['bike_out']: # ['subway_in']: # ['subway_out']:
        # for contextual_dataset_names in [['subway_in','bike_in','bike_out'],['subway_in','bike_out']]: #[ ['subway_in','bike_in'],['subway_in'],['bike_in'],[],['bike_in','bike_out'] ]:
        # for contextual_dataset_names in [['subway_out','bike_in','bike_out'],['subway_out','bike_out'], ['subway_out','bike_in'],['subway_out'],['bike_in'],['bike_out'],['bike_in','bike_out'] ]:
        for contextual_dataset_names in [['subway_in','subway_out','weather'],
                                         ['subway_in','weather'],
                                         ['subway_out','weather'],
                                         ['subway_in'],
                                         ['subway_out'],
                                         ['weather'],
                                         [],]: #[['subway_in','subway_out','weather'],['subway_in','subway_out'],['subway_in'],['subway_out'],['subway_in','weather'],['subway_out','weather'],[]]:  # ['subway_in'],['weather','subway_in'],[],['weather'],
            if 'weather'in contextual_dataset_names:
                possible_contextual_kwargs['weather'] = weather_dict_config[weather_method]
            # for horizon in [1,2,3,4]:
            for horizon in [1,2,4]: #[1,2]:
                for n_bis in range(1,6): # range(1,6): # range(1,6):
                    dataset_names =  [target_data] +contextual_dataset_names+ ['calendar']
                    name_i = f"{'_'.join(dataset_names)}"
                    name_i_end = f"_Agg100_Freq15min_e100_h{horizon}_bis{n_bis}"
                    if 'subway_in' in contextual_dataset_names or 'subway_out' in contextual_dataset_names:
                        name_i = f"{name_i}_ConcatLateAttnH4L3D48FF128"
                    if 'weather'in contextual_dataset_names:
                        name_i = f"{name_i}_{weather_method}"
                    name_i = f"{name_i}_{name_i_end}"
                        

                    config_i =  {'target_data': target_data,
                                    'dataset_names': dataset_names,
                                    'dataset_for_coverage': ['subway_in'],  # ['subway_in','netmob_POIs'],
                                    'calendar_types':['dayofweek', 'timeofday'],

                                    'input_embedding_dim': 48, # 24
                                    'adaptive_embedding_dim': 32,
                                    'tod_embedding_dim': 6,
                                    'dow_embedding_dim': 6,
                                    'feed_forward_dim': 256,
                                    'num_heads': 4,
                                    'num_layers': 3,

                                    'use_mixed_proj': True,
                                    'freq': '15min',
                                    'H':6,
                                    'D':1,
                                    'W':0,

                                    'lr': 0.001,
                                    'weight_decay':  0.0015,
                                    'dropout': 0.2,
                                    'torch_scheduler_milestone': 20,
                                    'torch_scheduler_gamma':0.9925,
                                    'torch_scheduler_type': 'warmup',
                                    'torch_scheduler_lr_start_factor': 0.3,
                                    'standardize': False,
                                    'minmaxnorm': True,
                                    'batch_size': 128,
                                    'epochs':300,

                                    'horizon_step': horizon,
                                    'step_ahead': horizon,

                                    'target_kwargs' : {target_data: possible_target_kwargs[target_data]},
                                    'contextual_kwargs' : {ds_name:possible_contextual_kwargs[ds_name] for ds_name in contextual_dataset_names },  
                                    'denoising_names':[],
                                    }  

                    modifications[name_i] = config_i


if __name__ == "__main__":

    target_data = 'bike_out' # 'PeMS08_flow' # 'CRITER_3_4_5_lanes_flow' #'subway_in'  # PeMS03 # PeMS04 # PeMS07 # PeMS08 # METR_LA # criter
    model_name = 'STAEformer'
    loger = LOG()


    SEED = 1
    modification_init = {}
    set_seed(SEED)

    compilation_modification = {#'epochs' : 1, #100
                                'SEED' : SEED, 
                                'num_workers' : 4, # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
                                'persistent_workers' : True ,# False 
                                'pin_memory' : True ,# False 
                                'prefetch_factor' : 4, # None, 2,3,4,5 ... 
                                'drop_last' : False,  # True
                                'mixed_precision' : False, # True # False
                                'torch_compile' : 'compile', # 'compile',# 'compile', #'compile' # 'jit_script' #'trace' # False
                                'loss_function_type':'HuberLoss',
                                'optimizer': 'adamw',
                                'unormalize_loss' : True,
                                'use_target_as_context':False,
                                'device': torch.device('cuda:0')
                                }
    

    log_final  = f"\n--------- Resume ---------\n"
    subfolder = f'{target_data}_{model_name}'
    # subfolder = f'{target_data}_{model_name}_coverage_NetMob'
    for trial_id,modification_i in modifications.items():
        print('\n>>>>>>>>>>>> TRIAL ID:',trial_id)
        config = modification_init.copy()
        config.update(compilation_modification)
        config.update(modification_i)

        args_init = local_get_args(model_name,
                        args_init = None,
                        dataset_names=config['dataset_names'],
                        dataset_for_coverage=config['dataset_for_coverage'],
                        modification = config)
        fold_to_evaluate=[args_init.K_fold-1]



        # Run the script
        weights_save_folder = f"K_fold_validation/training_wo_HP_tuning/optim/{subfolder}"
        weights_save_folder_w_root = f"{SAVE_DIRECTORY}/{weights_save_folder}"
        if not os.path.exists(weights_save_folder_w_root):
            os.mkdir(weights_save_folder_w_root)

        save_folder = f"{weights_save_folder}/{trial_id}"
        save_folder_with_root = f"{SAVE_DIRECTORY}/{save_folder}"
        print(f"Save folder: {save_folder_with_root}")
        if not os.path.exists(save_folder_with_root):
            os.mkdir(save_folder_with_root)
            
        # Train Model
        trainer,ds,model,args = model_loading_and_training(fold_to_evaluate,
                                        save_folder = weights_save_folder,
                                        args_init = args_init,
                                        modification =config,
                                        trial_id = trial_id)
    

        condition1,condition2,fold = get_conditions(args,fold_to_evaluate,[ds])
        valid_losses,df_loss,training_mode_list,metric_list,dic_results= init_metrics(args)
        df_loss, valid_losses,dic_results = keep_track_on_metrics(trainer,args,df_loss,valid_losses,dic_results,fold_to_evaluate,fold,condition1,condition2,training_mode_list,metric_list)

        save_model_metrics(trainer,args,valid_losses,training_mode_list,metric_list,df_loss,dic_results,save_folder,trial_id)
        test_metrics = trainer.performance['test_metrics']

        loger.add_log(test_metrics,['rmse','mae','mape','mse'],trial_id, args.step_ahead,args.horizon_step)
        
    loger.display_log()