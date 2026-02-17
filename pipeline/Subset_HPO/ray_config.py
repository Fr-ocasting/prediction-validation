import pkg_resources
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import torch
import importlib

import os


def get_scheduler(HP_max_epochs,name='ASHA', metric= 'Loss_model', mode = 'min',grace_period=2):
    if HP_max_epochs<=grace_period:
         grace_period = HP_max_epochs

    if name == 'ASHA':
        scheduler = ASHAScheduler(metric=metric,
            mode=mode,
            max_t=HP_max_epochs,  # Maximum of run epochs 
            grace_period=grace_period,     # Minimum of run epochs 
            reduction_factor=2,  # 100*(1/reduction_factor) % of all trials are kept each time they are reduced
        )
    elif name is None:
        scheduler = None
    else:
        raise NotImplementedError(f'Scheduler {name} has not been implemented' ) 
    
    return(scheduler)


def get_search_alg(name,metric,mode,points_to_evaluate = None):
    if name=='HyperOpt':
         search_alg = HyperOptSearch(metric = metric,
                                     mode = mode,
                                     #points_to_evaluate = points_to_evaluate
                                     )

    elif name is None:
        search_alg = None
    else:
        raise NotImplementedError(f'Scheduler {name} has not been implemented' ) 
    
    return(search_alg)

def get_point_to_evaluate(args):
    confif_path = f"pipeline.Flex_MDI.dl_models.{args.model_name}.load_config"
    default_config_model = importlib.import_module(confif_path)
    importlib.reload(default_config_model)
    dic_args_model = vars(default_config_model.args)
    dic_args__HP_model = vars(default_config_model.args_HP)
    point_to_evaluate = [dic_args_model.update(dic_args__HP_model)]

    return(point_to_evaluate)



def choose_ray_metric():
    ray_version = pkg_resources.get_distribution("ray").version
    if ray_version.startswith('2.7') or ray_version.startswith('3') :
        metric = 'Loss_model'
    else:
        metric = '_metric/Loss_model'
    return(metric)


def get_ray_config(args):

    # IF RESTRICT TO ONLY ONE GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_fraction = 0.5 
    metric = choose_ray_metric()
    points_to_evaluate = get_point_to_evaluate(args)   

    scheduler = get_scheduler(args.HP_max_epochs,args.ray_scheduler, metric= metric, mode = 'min',grace_period = args.grace_period)
    search_alg = get_search_alg(args.ray_search_alg,metric= metric,mode = 'min',points_to_evaluate = points_to_evaluate)

    resources_per_trial = {'gpu':gpu_fraction,'cpu':6} if torch.cuda.is_available() else {'cpu':1}
    num_gpus = available_gpus 
    num_cpus = 24 if torch.cuda.is_available() else 6
    max_concurrent_trials = int(available_gpus / gpu_fraction) if available_gpus > 0 else 6

    return(scheduler,search_alg,resources_per_trial,num_gpus,max_concurrent_trials,num_cpus)
    