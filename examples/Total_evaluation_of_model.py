import pandas as pd  # if not, I get this error while running a .py from terminal: 
# ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found (required by /root/anaconda3/envs/pytorch-2.0.1_py-3.10.5/lib/python3.10/site-packages/pandas/_libs/window/aggregations.cpython-310-x86_64-linux-gnu.so)

# Relative path:
import sys 
import os 
current_path = os.path.abspath(os.path.dirname(__file__))
working_dir = os.path.abspath(os.path.join(current_path,'..'))
#parent_dir = f"{parent_dir}/prediction_validation/"  # A utiliser sur .ipynb notebook
if working_dir not in sys.path:
    sys.path.insert(0,working_dir)
# ...

# Personnal import 
from examples.HP_parameter_choice import hyperparameter_tuning
from examples.train_model_on_k_fold_validation import train_model_on_k_fold_validation



def HP_and_valid_one_config(args,epochs_validation,num_samples):
    # HP Tuning on the first fold
    analysis,trial_id = hyperparameter_tuning(args,num_samples)

    # K-fold validation with best config: 
    train_model_on_k_fold_validation(trial_id,load_config=True,save_folder='K_fold_validation/training_with_HP_tuning',epochs=epochs_validation,hp_tuning_on_first_fold = True)



if __name__ == '__main__':

    #from file00 import *
    #vision_model_name = 'FeatureExtractorEncoderDecoder'  # 'ImageAvgPooling'  # 'FeatureExtractor_ResNetInspired_bis'  #'FeatureExtractor_ResNetInspired' #'MinimalFeatureExtractor',
    # 'AttentionFeatureExtractor' # 'FeatureExtractorEncoderDecoder' # 'VideoFeatureExtractorWithSpatialTemporalAttention'
    from examples.benchmark import local_get_args

    model_name = 'STGCN' #'CNN'
    dataset_for_coverage = ['subway_in','netmob_POIs'] 
    if False:
        model_name = 'STGCN' #'CNN'
        dataset_for_coverage = ['subway_in','netmob_POIs'] 
        dataset_names = ['calendar']
        vision_model_name = None

        args,_,_ = local_get_args(model_name,
                                args_init = None,
                                dataset_names=dataset_names,
                                dataset_for_coverage=dataset_for_coverage,
                                modification = {'ray':True,
                                                'grace_period':10,
                                                'HP_max_epochs':100,
                                                'evaluate_complete_ds' : True,
                                                'set_spatial_units' : ['BON','SOI','GER','CHA'],
                                                'vision_model_name': None
                                                })

        # Init 
        epochs_validation = 30
        num_samples = 1000
        
        HP_and_valid_one_config(args,epochs_validation,num_samples)
    if True:
        for dataset_names,vision_model_name in zip([['subway_in']], #['subway_in','subway_out']
                                                   [None]): #'VariableSelectionNetwork'
            args,_,_ = local_get_args(model_name,
                                    args_init = None,
                                    dataset_names=dataset_names,
                                    dataset_for_coverage=dataset_for_coverage,
                                    modification = {'ray':True,
                                                    'grace_period':20,
                                                    'HP_max_epochs':100,
                                                    'evaluate_complete_ds' : True,
                                                    'vision_model_name': vision_model_name,
                                                    #'vision_concatenation_early':True,   
                                                    #'vision_concatenation_late':True,
                                                    #'vision_num_heads':4
                                                   }
                                    
                                     )

            # Init 
            epochs_validation = 100
            num_samples = 500

            # HP and evaluate K-fold best config
            HP_and_valid_one_config(args,epochs_validation,num_samples)