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
from constants.config_by_datasets import dic_config

# Init:
#['subway_indiv','tramway_indiv','bus_indiv','velov','criter']
target_data = 'subway_in' #'subway_in'  # PeMS03 # PeMS04 # PeMS07 # PeMS08 # METR_LA 
dataset_names = ['subway_in','calendar_embedding'] #['PeMS03'] #['subway_in'] ['subway_in','subway_indiv'] #["subway_in","subway_out"] # ['subway_in','netmob_POIs_per_station'],["subway_in","subway_out"],["subway_in","calendar"] # ["subway_in"] # ['data_bidon'] # ['METR_LA'] # ['PEMS_BAY']
dataset_for_coverage = ['subway_in','netmob_image_per_station']#['subway_in','subway_indiv'] # ['subway_in','netmob_image_per_station'] #  ['data_bidon','netmob'] #  ['subway_in','netmob']  # ['METR_LA'] # ['PEMS_BAY']
model_name = 'STGCN' # 'STGCN', 'ASTGCN' # 'STGformer' #'STAEformer' # 'DSTRformer'
#station = ['BEL','PAR','AMP','SAN','FLA']# ['BEL','PAR','AMP','SAN','FLA']   # 'BON'  #'GER'
# ...

# Modif 
modification = {'target_data': target_data, 
                    'freq': '15min', #  '5min', 
                    'step_ahead': 4,
                    'use_target_as_context': False,
                    'data_augmentation': False,
            
                    'epochs' : 1, #100

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
                    'denoising_names':['subway_in','subway_out'],
                    'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                    'denoising_modes':["train","valid","test"],             # par défaut
                    'denoiser_kwargs':{'exponential': {'alpha': 0.8}}, # {'savitzky_golay': {'window': 5, 'poly': 2}} # {'exponential': {'alpha':0.3}} # {"median": {"kernel_size": 2}}

                    #
                    #'graph_conv_type': 'graph_conv', # 'cheb_graph_conv', 'graph_conv'
                    #'learnable_adj_top_k': 10,
                    #'learnable_adj_embd_dim': 16, 
                    ### ========

                    ### Time Embedding parameters:
                    'TE_concatenation_early': False, # True # False
                    'TE_concatenation_late': True, # True # False

                    ### Temporal Graph Transfermer Encoder parametrs:
                    #'TGE_num_layers' : 4, #2
                    #'TGE_num_heads' :  1, #IMPOSSIBLE > 1 CAR DOIT DIVISER L = 7
                    #'TGE_FC_hdim' :  32, #32

                    ### Netmob Parametrs: 
                    #'NetMob_only_epsilon': True,    # True # False
                    #'NetMob_selected_apps': ['Apple_iMessage','Web_Ads'],# ['Apple_iMessage','Web_Ads'], #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
                    #'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                    #'NetMob_selected_tags' : ['station_epsilon100'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                    #'NetMob_expanded' : '', # '' # '_expanded

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

                            "mlp_ratio": 1.5, # choices = [1.0, 1.5, 2.0, 2.5, 3.0]  # PAS SUR 
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
    })

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
                            "tod_embedding_dim": 4, # choices = [0, 4, 8, 12, 16]
                            "dow_embedding_dim": 4, # choices = [0, 4, 8, 12, 16]
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

modification.update({'num_workers' : 4, # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
                        'persistent_workers' : True ,# False 
                        'pin_memory' : True ,# False 
                        'prefetch_factor' : 4, # None, 2,3,4,5 ... 
                        'drop_last' : False,  # True
                        'mixed_precision' : False, # True # False
                        'torch_compile' : False, # 'compile' # 'jit_sript'
    })

if target_data in dic_config.keys():
    modification.update(dic_config[target_data])


args_init = local_get_args(model_name,
                    args_init = None,
                    dataset_names=dataset_names,
                    dataset_for_coverage=dataset_for_coverage,
                    modification = modification)


def main():
    ds,args,trial_id,save_folder,df_loss = get_ds(modification=modification,args_init=args_init)
    model = full_model(ds, args).to(args.device)
    optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
    trainer = Trainer(ds,model,args,optimizer,loss_function,scheduler = scheduler,show_figure = False,trial_id = trial_id, fold=0,save_folder = save_folder)
    trainer.train_and_valid(normalizer = ds.normalizer, mod = 1000,mod_plot = None) 
    return trainer,ds,model,args


if __name__ == "__main__":
    import numpy as np 
    import random 
    SEED = 42
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(SEED)

    # Run the script
    trainer,ds,model,args = main()
    print("Training completed successfully.")
    print(trainer.performance)