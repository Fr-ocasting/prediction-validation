import torch
import ray 
from ray import tune 

# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal imports: 
from utils.utilities_DL import get_MultiModel_loss_args_emb_opts,load_init_trainer,load_prediction
from DL_class import Trainer
from HP_tuning.ray_search_space import get_search_space_ray 
from HP_tuning.ray_config import get_ray_config
from constants.config import get_args
from constants.paths import folder_path,file_name


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def modify_args(args,name_gpu='cuda'):
    # Modification :
    args.epochs = 500
    args.K_fold = 6   # Means we will use the first fold for the Ray Tuning and the 5 other ones to get the metrics
    args.single_station = False
    args.ray = True
    
    if torch.cuda.is_available():
        args.device = name_gpu
        args.batch_size = 256
    else :
        args.device = 'cpu'
        args.batch_size = 32

    if args.loss_function_type == 'MSE':
        args.out_dim = 1
        args.alpha = None
        args.track_pi = False

    else:
        args.out_dim = 2
        args.alpha = 0.1
        args.track_pi = True

    print("!!! Prediction sur une UNIQUE STATION et non pas les 40 ") if args.single_station else None
    print(f"!!! Loss function: {args.loss_function_type} ") 
    print(f"Model: {args.model_name}, K_fold = {args.K_fold}") 
    
    return(args)    
        
## Hyper Parameter Tuning sur le Fold 0
def load_trainer(config,folder_path,file_name,args):

    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    Datasets,DataLoader_list,dic_class2rpz,nb_words_embedding,time_slots_labels,dic_rpz2class = load_init_trainer(folder_path,file_name,args)
    (loss_function,Model_list,Optimizer_list,Scheduler_list,args_embedding) = get_MultiModel_loss_args_emb_opts(args,nb_words_embedding,dic_class2rpz,n_vertex = len(Datasets[0].columns))
    dataset,dataloader,model,optimizer,scheduler = Datasets[0],DataLoader_list[0],Model_list[0],Optimizer_list[0],Scheduler_list[0]


    trainer = Trainer(dataset,model,dataloader,
                    args,optimizer,loss_function,scheduler = scheduler,
                    args_embedding=args_embedding,dic_class2rpz=dic_class2rpz)
    return(trainer)

def Train_with_tune(config,folder_path,file_name,args):
    trainer = load_trainer(config,folder_path,file_name,args)
    result_df = trainer.train_and_valid()

    
def run_tuning_and_save_results(args,num_samples):
    config = get_search_space_ray(args)
    ray_scheduler,ray_search_alg,resources_per_trial,num_gpus,max_concurrent_trials,num_cpus = get_ray_config(args)
    
    def trainer(config):
        return(Train_with_tune(config,folder_path,file_name,args))
        
    if ray.is_initialized:
        ray.shutdown()
        ray.init(num_gpus=num_gpus,num_cpus=num_cpus)

        
    analysis = tune.run(
            trainer,
            config=config,
            num_samples=num_samples,  # Increase num_samples for more random combinations
            resources_per_trial = resources_per_trial,
            max_concurrent_trials = max_concurrent_trials,
            scheduler = ray_scheduler,
            search_alg = ray_search_alg,
        )

    name_save = f"HyperparameterTuning/Htuning_ray_analysis_{args.model_name}_loss{args.loss_function_type}_TE_{args.time_embedding}"
    analysis.results_df.to_csv(f'{name_save}.csv')
    
    

if __name__ == '__main__': 
    
    # ==== GET PARAMETERS ====
    # Load config
    model_name = 'STGCN'  #'CNN'
    args = get_args(model_name)
    # Modification pour HyperParameterTuning:
    args.time_embedding = True
    args.loss_function_type =  'MSE' #'quantile' 
    #args = get_args(model_name = model_name,learn_graph_structure = True)  # MTGNN

    args = modify_args(args,name_gpu='cuda:0')
    print('abs_path: ',args.abs_path)
    run_tuning_and_save_results(args,num_samples=1000)