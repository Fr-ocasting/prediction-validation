from ray import tune
import importlib
import sys
import os

# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def get_search_space_ray(args):
    # Common search space
    config = {"lr": tune.loguniform(5e-5, 5e-3), # tune.qloguniform(1e-5, 1e-2, 1e-5), # tune.qloguniform(5e-5, 5e-3, 5e-5)
              "weight_decay" : tune.loguniform(5e-4, 1e-2), #tune.uniform(0.03, 0.095),# tune.uniform(0.0005, 0.1)
              #"momentum" : tune.uniform(0.80, 0.99),
              "dropout" : tune.uniform(0.0,0.7), # tune.choice([0,0.05,0.1,0.15,0.2,0.7]),
            #   "scheduler" :  {"torch_scheduler_milestone": tune.choice([1,5,10,20,50]), "torch_scheduler_gamma": tune.uniform(0.98, 0.995), "torch_scheduler_lr_start_factor": tune.uniform(0.3, 0.9)}#{"torch_scheduler_milestone": tune.choice([1,5,10,20]), "torch_scheduler_gamma": tune.uniform(0.98, 0.999), "torch_scheduler_lr_start_factor": tune.uniform(0.2, 0.8)}
              #tune.choice([{'scheduler':True, "torch_scheduler_milestone": tune.randint(1, 20),
              #                "torch_scheduler_gamma": tune.uniform(0.985, 0.999), "torch_scheduler_lr_start_factor": tune.uniform(0.1, 1)},
              #                  {'scheduler':None }])
              }

    #if 'netmob_POIs' in args.dataset_name:
    #    config.update({'epsilon_clustering':tune.uniform(0.01, 0.4)})
    
    # Load Search Space associated to the Model: 
    module_path = f"dl_models.{args.model_name}.search_space"
    search_space_module = importlib.import_module(module_path)
    importlib.reload(search_space_module)
    globals()[f"config_{args.model_name}"] = search_space_module.config

    # Update Config : 
    config.update(globals()[f"config_{args.model_name}"])

   # Tackle Embedding
    if len(vars(args.args_embedding))>0:
        module_path = f"dl_models.TimeEmbedding.search_space"
        search_space_module = importlib.import_module(module_path)
        importlib.reload(search_space_module)
        config_TE= search_space_module.config  

        keys = list(config_TE.keys())
        # Add specific 'TE' argument for time embedding
        for key in keys:
            config_TE[f"TE_{key}"] = config_TE[key]
            config_TE.pop(key) 

        config.update(config_TE)

    # Tackle Vision Models
    if hasattr(args, 'args_vision') and len(vars(args.args_vision)) > 0:
        module_path = f"dl_models.vision_models.{args.args_vision.model_name}.search_space"
        search_space_module = importlib.import_module(module_path)
        importlib.reload(search_space_module)
        config_vision = search_space_module.config  

        keys = list(config_vision.keys())
        for key in keys:
            config_vision[f"vision_{key}"] = config_vision[key]
            config_vision.pop(key) 

        config.update(config_vision)

    return(config)    