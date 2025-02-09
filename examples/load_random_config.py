import sys
import os
import numpy as np 
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from examples.benchmark import local_get_args

default_args = dict(model_name='STGCN',
                    args_init = None,
                    dataset_names=['subway_in'],
                    dataset_for_coverage=['subway_in','netmob_POIs'],
                    modification = {},
                    vision_model_name = None
                    )
                            

def get_default_args(modification):
    if modification is not None:
        kwargs = {}
        for attr in ['model_name','dataset_for_coverage','dataset_names']:
            if attr in modification.keys():
                kwargs[attr] =  modification[attr]
                del modification[attr]
            else:
                kwargs[attr] =  default_args[attr]     
        kwargs['modification'] = modification['modification']
    else:
        kwargs = default_args
        
    kwargs['args_init'] = None 
    
    args = local_get_args(**kwargs)

    if hasattr(kwargs['modification'],'fold_to_evaluate'):
        folds = kwargs['modification']['fold_to_evaluate']
    else:
        folds = list(np.arange(args.K_fold))
    return args,folds