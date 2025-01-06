import sys
import os
import numpy as np 
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from examples.benchmark import local_get_args

model_name = 'STGCN' #'CNN'
dataset_for_coverage = ['subway_in','netmob_POIs'] 
dataset_names = ['subway_in']
vision_model_name = [None]
args,_,_ = local_get_args(model_name,
                        args_init = None,
                        dataset_names=dataset_names,
                        dataset_for_coverage=dataset_for_coverage,
                        modification = {'evaluate_complete_ds' : True,
                                        'vision_model_name': vision_model_name,
                                        
                                        'lr':1e-3,
                                        'weight_decay':0.05,
                                        'dropout':0.8,
                                        'scheduler':True,
                                        'torch_scheduler_milestone':5,
                                        'torch_scheduler_gamma':0.997,
                                        'torch_scheduler_lr_start_factor':1,
                                        #'vision_concatenation_early':True,   
                                        #'vision_concatenation_late':True,
                                        #'vision_num_heads':4
                                        }
                            )
folds = list(np.arange(args.K_fold))