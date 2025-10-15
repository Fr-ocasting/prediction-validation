import torch

SEED = 1

compilation_modification = {'SEED' : SEED, 
                            'num_workers' : 4, # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
                            'persistent_workers' : False ,# False 
                            'pin_memory' : True ,# False 
                            'prefetch_factor' : 4, # None, 2,3,4,5 ... 
                            'drop_last' : False,  # True
                            'mixed_precision' : False, # True # False
                            'torch_compile' :  'compile', # 'compile', #'compile',# 'compile',# 'compile', #'compile' # 'jit_script' #'trace' # False
                            'loss_function_type':'HuberLoss',
                            'optimizer': 'adamw',
                            'unormalize_loss' : True,
                            'use_target_as_context':False,

                            'device': torch.device('cuda:0')
    }


assert not(compilation_modification['persistent_workers'] and compilation_modification['pin_memory']), "persistent_workers and pin_memory cannot be both True, it might result with 'RuntimeError: Pin memory thread exited unexpectedly'"