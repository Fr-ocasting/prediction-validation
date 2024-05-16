from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import torch

def get_scheduler(epochs,name='ASHA', metric= 'Loss_model', mode = 'min'):
    if name == 'ASHA':
        scheduler = ASHAScheduler(metric=metric,
            mode=mode,
            max_t=epochs,  # Maximum of run epochs 
            grace_period=1,     # Minimum of run epochs 
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
                                     points_to_evaluate = points_to_evaluate
                                     )

    elif name is None:
        search_alg = None
    else:
        raise NotImplementedError(f'Scheduler {name} has not been implemented' ) 
    
    return(search_alg)





def get_ray_config(args):
    #metric = '_metric/Loss_model' if args.ray_track_pi else 'Loss_model'
    metric = '_metric/Loss_model'
    
    scheduler = get_scheduler(args.epochs,args.ray_scheduler, metric= metric, mode = 'min')
    search_alg = get_search_alg(args.ray_search_alg,metric= metric,mode = 'min',points_to_evaluate = None)

    resources_per_trial = {'gpu':0.25,'cpu':2} if torch.cuda.is_available() else {'cpu':1}
    num_gpus = 1 if torch.cuda.is_available() else 0
    max_concurrent_trials = 18 if torch.cuda.is_available() else 6


    return(scheduler,search_alg,resources_per_trial,num_gpus,max_concurrent_trials)
    