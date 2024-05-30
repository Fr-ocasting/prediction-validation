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

    if args.time_embedding:
        config.update(config_embedding)

    if args.model_name == 'STGCN':
         config.update(config_stgcn)
            
    return(config)    