from ray import tune

def get_search_space_ray(args):
    
    config = {"lr": tune.qloguniform(1e-4, 1e-1, 5e-5),
              "weight_decay" : tune.uniform(0.0005, 0.1),
              "momentum" : tune.uniform(0.80, 0.99),
              "dropout" : tune.uniform(0,0.9),
              "scheduler" : tune.choice([True,None]),

              "torch_scheduler_milestone": tune.qrandint(10, 300, 4),
              "torch_scheduler_gamma": tune.uniform(0.985, 0.999),
              "torch_scheduler_lr_start_factor": tune.uniform(0.1, 1), 
            }

    config_embedding = {#'calendar_class' : tune.choice([1,2,3]),
                        'embedding_dim' : tune.choice([3,4,8]),
                        'multi_embedding' : tune.choice([True,False]),
                        #'TE_transfer' : tune.choice([True,False]),
                        }


    config_stgcn = {"Kt" : tune.choice([2,3,4]),
                    "stblock_num" : tune.choice([2,3,4]),
                    "act_fun" : tune.choice(['glu','gtu']),
                    "Ks" :  tune.choice([2,3]),
                    "graph_conv_type" : tune.choice(['cheb_graph_conv','graph_conv']),
                    "gso_type" : tune.choice(['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj']),
                    "adj_type" : tune.choice(['adj','corr','dist'])
                    }
    
    # Tackle Core Model:
    if args.model_name == 'STGCN':
         config.update(config_stgcn)

    # Tackle Embedding
    if args.time_embedding:
        config.update(config_embedding)

    # Tackle Vision Models
    if len(vars(args.args_vision)) > 0:
        # ImageAvgPooling
        if args.args_vision.model_name == 'ImageAvgPooling':
            config_vision = {}  # No possible HP Tuning
 
        elif args.args_vision.model_name == 'FeatureExtractor_ResNetInspired':
            config_vision = {'vision_h_dim': tune.choice([8,16,32,64])} #,64,128,256
             

        # MinimalFeatureExtractor  
        elif args.args_vision.model_name == 'MinimalFeatureExtractor':
            config_vision = {'vision_h_dim': tune.choice([8,16,32,64]) #,64,128,256
                             } 


        else:
            raise NotImplementedError(f"Model {args.args_vision.model_name} has not been implemented for HP Tuning")
        
        config.update(config_vision)




    return(config)    