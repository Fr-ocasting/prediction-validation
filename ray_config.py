import pkg_resources
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import torch

def get_scheduler(epochs,name='ASHA', metric= 'Loss_model', mode = 'min'):
    if name == 'ASHA':
        scheduler = ASHAScheduler(metric=metric,
            mode=mode,
            max_t=epochs,  # Maximum of run epochs 
            grace_period=15,     # Minimum of run epochs 
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

def get_point_to_evaluate(args):
    if args.model_name == 'STGCN':
        point_to_evaluate = [{
            'Kt': 3,
            'Ks': 2,
            'graph_conv_type': 'graph_conv',
            'gso_type': 'sym_norm_lap',
            'adj_type':'dist',
            'dropout': 0.2,
            'lr': 1e-4,
            'momentum':0.99,
            'weight_decay':0.005},
            {
            'Kt': 4,
            'Ks': 2,
            'graph_conv_type': 'graph_conv',
            'gso_type': 'rw_norm_lap',
            'adj_type':'dist',
            'dropout': 0.15,
            'lr': 4e-4,
            'momentum':0.87,
            'weight_decay':0.05,
            'stblock_num':3,
            'act_fun':'gtu'}
            ]
    elif args.model_name == 'CNN':
        point_to_evaluate = [{
            'c_in': 1,
            'C_outs': [16,2] ,
            'H_dims': [16,16]
        }]

    elif args.model_name == 'DCRNN':
        point_to_evaluate = [{
            'torch_scheduler_milestone' : 40,
            'torch_scheduler_gamma' :0.99,
            'torch_scheduler_lr_start_factor' : 0.1,
            

        }]

    else:
        raise NotImplementedError(f'Point to Evaluate of Ray Search Algorithm for {args.model_name} has not been implemented' ) 
    return(point_to_evaluate)



def choose_ray_metric(args):
    ray_version = pkg_resources.get_distribution("ray").version
    if ray_version.startswith('2.7'):
        metric = 'Loss_model'
    else:
        metric = '_metric/Loss_model'
    return(metric)


def get_ray_config(args):
    metric = choose_ray_metric(args)
    points_to_evaluate = get_point_to_evaluate(args)   

    scheduler = get_scheduler(args.epochs,args.ray_scheduler, metric= metric, mode = 'min')
    search_alg = get_search_alg(args.ray_search_alg,metric= metric,mode = 'min',points_to_evaluate = None)

    resources_per_trial = {'gpu':0.25,'cpu':2} if torch.cuda.is_available() else {'cpu':1}
    num_gpus = 2 if torch.cuda.is_available() else 0
    num_cpus = 36 if torch.cuda.is_available() else 6
    max_concurrent_trials = 18 if torch.cuda.is_available() else 6


    return(scheduler,search_alg,resources_per_trial,num_gpus,max_concurrent_trials,num_cpus)
    