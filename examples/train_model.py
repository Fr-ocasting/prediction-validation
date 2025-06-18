# GET PARAMETERS
import os 
import sys
import torch 
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True
# Get Parent folder : 

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from examples.benchmark import local_get_args
from examples.train_and_visu_non_recurrent import evaluate_config,train_the_config,get_ds
from high_level_DL_method import load_optimizer_and_scheduler
from dl_models.full_model import full_model
from trainer import Trainer
from utils.loger import LOG
from utils.rng import set_seed
loger = LOG()
# Init:
#['subway_indiv','tramway_indiv','bus_indiv','velov','criter']
target_data = 'PeMS08_flow'#'PeMS08_flow'#'CRITER_3_4_5_lanes_flow' #'subway_in'  # PeMS03 # PeMS04 # PeMS07 # PeMS08 # METR_LA # criter
dataset_names = ['PeMS08_flow'] # ['CRITER_3_4_5_lanes_flow','netmob_POIs']  #['CRITER_3_4_5_lanes_flow']#['PeMS08_flow','PeMS08_occupancy','PeMS08_speed'] # ['subway_in','calendar_embedding'] #['PeMS03'] #['subway_in'] ['subway_in','subway_indiv'] #["subway_in","subway_out"] # ['subway_in','netmob_POIs_per_station'],["subway_in","subway_out"],["subway_in","calendar"] # ["subway_in"] # ['data_bidon'] # ['METR_LA'] # ['PEMS_BAY']
dataset_for_coverage =['PeMS08_flow'] # ['CRITER_3_4_5_lanes_flow','netmob_POIs'] #['PeMS08_flow'] # ['subway_in','netmob_image_per_station']#['subway_in','subway_indiv'] # ['subway_in','netmob_image_per_station'] #  ['data_bidon','netmob'] #  ['subway_in','netmob']  # ['METR_LA'] # ['PEMS_BAY']
model_name = 'STAEformer' # 'STGCN', 'ASTGCN' # 'STGformer' #'STAEformer' # 'DSTRformer'
#station = ['BEL','PAR','AMP','SAN','FLA']# ['BEL','PAR','AMP','SAN','FLA']   # 'BON'  #'GER'
# ...

# Modif 
modification = {'target_data': target_data, 
                    'freq': '6min',#'5min', #  '15min', 
                    'step_ahead': 4, # 12, # 4
                    'use_target_as_context': False,
                    'data_augmentation': False,
            
                    'epochs' : 0, #100

                    'lr': 0.00105, # 5e-5,# 4e-4,
                    'weight_decay': 0.0188896655584368, # 0.05,
                    'dropout': 0.271795372610271, # 0.15,

                    'scheduler': True,  # None
                    'torch_scheduler_milestone': 28.0, #5,
                    'torch_scheduler_gamma': 0.9958348861339396, # 0.997,
                    'torch_scheduler_lr_start_factor': 0.8809942312067847, # 1,


                    #'set_spatial_units':  station,   

                    'stacked_contextual': True, # True # False
                    'temporal_graph_transformer_encoder': False,
                    'compute_node_attr_with_attn' : False, # True ??

                    ### Denoising: 
                    #'denoising_names':['subway_in','subway_out'],
                    #'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                    #'denoising_modes':["train","valid","test"],             # par défaut
                    #'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}

                    #
                    #'graph_conv_type': 'graph_conv', # 'cheb_graph_conv', 'graph_conv'
                    #'learnable_adj_top_k': 10,
                    #'learnable_adj_embd_dim': 16, 
                    ### ========

                    ### Time Embedding parameters:
                    #'TE_concatenation_early': False, # True # False
                    #'TE_concatenation_late': True, # True # False

                    ### Temporal Graph Transfermer Encoder parametrs:
                    #'TGE_num_layers' : 4, #2
                    #'TGE_num_heads' :  1, #IMPOSSIBLE > 1 CAR DOIT DIVISER L = 7
                    #'TGE_FC_hdim' :  32, #32

                    ### Netmob Parametrs: 
                    'NetMob_only_epsilon': False,    # True # False
                    'NetMob_selected_apps': ['Deezer','Web_Weather','Google_Maps'],# ['Apple_iMessage','Web_Ads'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                    'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                    'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                    'NetMob_expanded' : '', # '' # '_expanded

                    ### Compute node with attention parameters: 
                    #'vision_num_heads':6,
                    #"vision_grn_out_dim":48,
                    #'vision_model_name': 'VariableSelectionNetwork',
                    #'vision_concatenation_early':True,   
                    #'vision_concatenation_late':True,
                            }


if model_name == 'DSTRformer':
    dataset_names.append('calendar')
    modification.update({ "input_embedding_dim": 16, # choices = [16, 24, 32, 48, 64]
                            "tod_embedding_dim": 4, # choices = [0, 4, 8, 12, 16]
                            "dow_embedding_dim": 4, # choices = [0, 4, 8, 12, 16]
                            "num_heads": 2, # choices = [1, 2, 4, 8]  # Has to devide input_embedding_dim+tod_embedding_dim+dow_embedding_dim+adaptive_embedding_dim
                            "num_layers": 2, # choices = [1, 2, 3, 4, 6]

                            "adaptive_embedding_dim": 12, # choices = [8, 12, 16, 24, 32] # help = ' has to be < num_nodes.
                            "out_feed_forward_dim": 64, # choices = [8, 16, 32, 64, 128, 256]
                            "num_layers_m": 1, # choices = [1, 2, 3, 4, 6]
                            "ts_embedding_dim": 8, # choices = [0, 4, 8, 12, 16]
                            "time_embedding_dim": 0, # choices = [0, 4, 8, 12, 16]
                            "mlp_num_layers": 1, # choices = [1, 2, 3, 4, 6]
                            "feed_forward_dim": 16, # choices = [8, 16, 32, 64, 128, 256]
                            "use_mixed_proj": True, # choices = [True, False]
                            "adj_type": 'adj', # choices = ['adj','dist','corr']
                            "adj_normalize_method": 'doubletransition', # choices = ['normlap','scalap','symadj','transition','doubletransition','identity']
                            "threshold": 0.7, # choices = [0.5, 0.7, 0.9] # useless if adj_type = 'adj'
                            'calendar_types':['dayofweek', 'timeofday']})

if model_name == 'STGformer':
    dataset_names.append('calendar')
    modification.update({ "input_embedding_dim": 16, # choices = [16, 24, 32, 48, 64]
                            "tod_embedding_dim": 4, # choices = [0, 4, 8, 12, 16]
                            "dow_embedding_dim": 4, # choices = [0, 4, 8, 12, 16]
                            "adaptive_embedding_dim": 12, # choices = [8, 12, 16, 24, 32] # help = ' has to be < num_nodes.

                            # Attention
                            "num_heads": 6, # choices = [1, 2, 4, 8]  # Has to devide input_embedding_dim+tod_embedding_dim+dow_embedding_dim+adaptive_embedding_dim
                            "num_layers": 6, # choices = [1, 2, 3, 4, 6]
                            "mlp_ratio": 1.5, # choices = [1.0, 1.5, 2.0, 2.5, 3.0]  # PAS SUR 

                            # adaptive embedding dropout 
                            "dropout_a": 0.3, # choices = uniform(0.0, 0.5)

                            # Kernel sizes pour la projection temporelle
                            "kernel_size": [1], # choices = [[1], [3], [1, 3], [3, 5]]
    })

if model_name == 'STAEformer':
    dataset_names.append('calendar')
    modification.update({ #"input_embedding_dim": 16, # choices = [16, 24, 32, 48, 64]
                            #"tod_embedding_dim": 4, # choices = [0, 4, 8, 12, 16]
                            #"dow_embedding_dim": 4, # choices = [0, 4, 8, 12, 16]
                            #"adaptive_embedding_dim": 12, # choices = [8, 12, 16, 24, 32] # help = ' has to be < num_nodes.
                            #"spatial_embedding_dim": 8, # choices = [4, 8, 12, 16,32,64]

                            # Attention
                            #"num_heads": 2, # choices = [1, 2, 4, 8]  # Has to devide input_embedding_dim+tod_embedding_dim+dow_embedding_dim+adaptive_embedding_dim
                            #"num_layers": 2, # choices = [1, 2, 3, 4, 6]
                            #"feed_forward_dim": 16,
    })

if model_name == 'STGCN':
    modification.update({'Kt': 2,
                        'stblock_num': 3,
                        'gso_type': 'sym_renorm_adj',
                        'temporal_h_dim': 256,
                        'spatial_h_dim': 32,
                        'output_h_dim': 16,
                        'adj_type':'corr',
                        'threshold': 0.7,
                        'learnable_adj_matrix' : False, # True                              # EXIST ONLY IF MODEL = STGCN
                        'graph_conv_type': 'graph_conv', # 'cheb_graph_conv', 'graph_conv'  # EXIST ONLY IF MODEL = STGCN
                        'learnable_adj_top_k': 10,                                          # EXIST ONLY IF MODEL = STGCN
                        'learnable_adj_embd_dim': 16,                                       # EXIST ONLY IF MODEL = STGCN  
                        })


def main(fold_to_evaluate,save_folder,args_init,modification,trial_id):
    ds,args,_,_,_ = get_ds(modification=modification,args_init=args_init,fold_to_evaluate=fold_to_evaluate)
    for key,value in vars(args).items():
        print(f"{key}: {value}")
    model = full_model(ds, args).to(args.device)
    optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
    if len(fold_to_evaluate) == 1: 
        fold = fold_to_evaluate[0]
    else:
        raise ValueError("fold_to_evaluate should contain only one fold cause only one training will be done here.")
    trainer = Trainer(ds,model,args,optimizer,loss_function,scheduler = scheduler,show_figure = False,trial_id = trial_id, fold=fold,save_folder = save_folder)
    trainer.train_and_valid(normalizer = ds.normalizer, mod = 1000,mod_plot = None,unormalize_loss = args.unormalize_loss) 
    return trainer,ds,model,args

if __name__ == "__main__":
    import numpy as np 
    import random 
    from constants.paths import SAVE_DIRECTORY
    from examples.train_model_on_k_fold_validation import save_model_metrics,get_conditions,keep_track_on_metrics,init_metrics
    import importlib
    

    config_file = importlib.import_module(f"constants.config_by_datasets.{target_data}.{model_name}")
    importlib.reload(config_file)
    modification = config_file.config
    SEED = config_file.SEED


    modification.update({'num_workers' : 4, # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
                        'persistent_workers' : True ,# False 
                        'pin_memory' : True ,# False 
                        'prefetch_factor' : 4, # None, 2,3,4,5 ... 
                        'drop_last' : False,  # True
                        'mixed_precision' : False, # True # False
                        'torch_compile' : False, # 'compile' # 'jit_script' #'trace' # False
                        'device': torch.device('cuda:0')
    })


    args_init = local_get_args(model_name,
                    args_init = None,
                    dataset_names=dataset_names,
                    dataset_for_coverage=dataset_for_coverage,
                    modification = modification)

    set_seed(SEED)


    # Run the script
    fold_to_evaluate=[args_init.K_fold-1]

    weights_save_folder = f"K_fold_validation/training_wo_HP_tuning"
    trial_id = f"{args_init.model_name}_{'_'.join(args_init.dataset_names)}_fold_{str(fold_to_evaluate[0])}_epochs_{args_init.epochs}"
    save_folder = f"{weights_save_folder}/{trial_id}"
    save_folder_with_root = f"{os.path.expanduser('~')}/prediction-validation/{SAVE_DIRECTORY}/{save_folder}"

    print(f"Save folder: {save_folder_with_root}")
    if not os.path.exists(save_folder_with_root):
        os.makedirs(save_folder_with_root)



    trainer,ds,model,args = main(fold_to_evaluate,weights_save_folder,args_init,modification)

    condition1,condition2,fold = get_conditions(args,fold_to_evaluate,[ds])
    valid_losses,df_loss,training_mode_list,metric_list,dic_results= init_metrics(args)
    df_loss, valid_losses,dic_results = keep_track_on_metrics(trainer,args,df_loss,valid_losses,dic_results,fold_to_evaluate,fold,condition1,condition2,training_mode_list,metric_list)

    save_model_metrics(trainer,args,valid_losses,training_mode_list,metric_list,df_loss,dic_results,save_folder,trial_id)

    test_metrics = trainer.performance['test_metrics']
    loger.add_log(test_metrics, ['rmse','mae','mape','mse'], trial_id, args.step_ahead,args.horizon_step)

    # log_final_i = f"All Steps RMSE = {'{:.2f}'.format(test_metrics['rmse_all'])}, MAE = {'{:.2f}'.format(test_metrics['mae_all'])}, MAPE = {'{:.2f}'.format(test_metrics['mape_all'])}, MSE = {'{:.2f}'.format(test_metrics['mse_all'])}"
    # log_final = log_final + f"{trial_id}:   {log_final_i}\n"
    # print(f"\n--------- Test ---------\n{log_final_i}")
    # for h in np.arange(1,args.step_ahead+1):
    #     print(f"Step {h} RMSE = {'{:.2f}'.format(test_metrics[f'rmse_h{h}'])}, MAE = {'{:.2f}'.format(test_metrics[f'mae_h{h}'])}, MAPE = {'{:.2f}'.format(test_metrics[f'mape_h{h}'])}, MSE = {'{:.2f}'.format(test_metrics[f'mse_h{h}'])}")



loger.display_log()
#print(log_final)