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
    config = {"lr": tune.qloguniform(1e-4, 1e-1, 5e-5),
              "weight_decay" : tune.uniform(0.0005, 0.1),
              #"momentum" : tune.uniform(0.80, 0.99),
              "dropout" : tune.uniform(0,0.9),
              "scheduler" : tune.choice([{'scheduler':True,
                                            "torch_scheduler_milestone": tune.quniform(1, 100, 1),
                                            "torch_scheduler_gamma": tune.uniform(0.985, 0.999),
                                            "torch_scheduler_lr_start_factor": tune.uniform(0.1, 1), 
                                        },
                                        {'scheduler':None
                                        }]
                                        )
              }
    
    # Load Search Space associated to the Model: 
    module_path = f"dl_models.{args.model_name}.search_space"
    search_space_module = importlib.import_module(module_path)
    globals()[f"config_{args.model_name}"] = search_space_module.config

    # Update Config : 
    config.update(globals()[f"config_{args.model_name}"])



   # Tackle Embedding
    if args.time_embedding:
        config_embedding = {#'calendar_class' : tune.choice([1,2,3]),
                        'embedding_dim' : tune.choice([3,4,8]),
                        'multi_embedding' : tune.choice([True,False]),
                        #'TE_transfer' : tune.choice([True,False]),
                        }
        config.update(config_embedding)

    # Tackle Vision Models
    if len(vars(args.args_vision)) > 0:
        module_path = f"dl_models.vision_models.{args.args_vision.model_name}.search_space"
        search_space_module = importlib.import_module(module_path)
        config_vision = search_space_module.config  

        keys = list(config_vision.keys())
        for key in keys:
            config_vision[f"vision_{key}"] = config_vision[key]
            config_vision.pop(key) 

        config.update(config_vision)
        '''
        # ImageAvgPooling
        if args.args_vision.model_name == 'ImageAvgPooling':
            config_vision = {}  # No possible HP Tuning
 
        elif args.args_vision.model_name == 'FeatureExtractor_ResNetInspired':
            config_vision = {'vision_h_dim': tune.choice([8,16,32,64])} #,64,128,256

        elif args.args_vision.model_name == 'FeatureExtractorEncoderDecoder':  # (c_in=3, z_dim=64, N=40)
            config_vision = {'vision_out_dim': tune.choice([8,16,32,64,128])
                             }

        elif args.args_vision.model_name == 'AttentionFeatureExtractor': # (c_in=3, z_dim=64, N=40)
            config_vision = {'vision_out_dim': tune.choice([8,16,32,64,128])
                             }

        elif args.args_vision.model_name == 'FeatureExtractor_ResNetInspired_bis':
            config_vision = {'vision_out_dim': tune.choice([8,16,32,64,128])
                             }

        elif args.args_vision.model_name == 'VideoFeatureExtractorWithSpatialTemporalAttention': # (c_in=3, out_dim=64, N=40, d_model=128))
            config_vision = {'vision_out_dim': tune.choice([8,16,32,64,128]),
                             'vision_d_model': tune.choice([8,16,32,64]),
                             } 
             
        # MinimalFeatureExtractor  
        elif args.args_vision.model_name == 'MinimalFeatureExtractor':
            config_vision = {'vision_h_dim': tune.choice([8,16,32,64]) #,64,128,256
                             } 
        else:
            raise NotImplementedError(f"Model {args.args_vision.model_name} has not been implemented for HP Tuning")
        '''
        
        




    return(config)    