from ray import tune

def get_search_space_ray(args):
    
    config = {"lr": tune.qloguniform(1e-4, 1e-1, 5e-5),
              "weight_decay" : tune.uniform(0.0005, 0.1),
              "momentum" : tune.uniform(0.80, 0.99),
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

    config_embedding = {#'calendar_class' : tune.choice([1,2,3]),
                    'embedding_dim' : tune.choice([3,4,8]),
                    'multi_embedding' : tune.choice([True,False]),
                    #'TE_transfer' : tune.choice([True,False]),
                    }

    config_stgcn = {"Kt" : tune.choice([2,3,4]),
                    "stblock_num" : tune.choice([2,3,4]),
                    "act_func" : tune.choice(['glu','gtu']),
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
        
        config.update(config_vision)




    return(config)    