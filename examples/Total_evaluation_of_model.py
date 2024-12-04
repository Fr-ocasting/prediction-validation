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



def HP_and_valid_one_config(args,epochs_validation,vision_model_name,num_samples):
    # HP Tuning on the first fold
    analysis,trial_id = hyperparameter_tuning(args,vision_model_name,num_samples)

    # K-fold validation with best config: 
    train_model_on_k_fold_validation(trial_id,load_config=True,save_folder='K_fold_validation/training_with_HP_tuning',epochs=epochs_validation,hp_tuning_on_first_fold = True)



if __name__ == '__main__':

    #from file00 import *
    #vision_model_name = 'FeatureExtractorEncoderDecoder'  # 'ImageAvgPooling'  # 'FeatureExtractor_ResNetInspired_bis'  #'FeatureExtractor_ResNetInspired' #'MinimalFeatureExtractor',
    # 'AttentionFeatureExtractor' # 'FeatureExtractorEncoderDecoder' # 'VideoFeatureExtractorWithSpatialTemporalAttention'
    from examples.benchmark import local_get_args

    model_name = 'STGCN' #'CNN'
    dataset_for_coverage = ['subway_in','netmob_POIs'] 
    if True:
        for dataset_names,vision_model_name in zip([['subway_in']],[None]):
            args,_,_ = local_get_args(model_name,
                                    args_init = None,
                                    dataset_names=dataset_names,
                                    dataset_for_coverage=dataset_for_coverage,
                                    modification = {'ray':True,
                                                    'grace_period':2,
                                                    'HP_max_epochs':30,
                                                    'evaluate_complete_ds' : True
                                                    })

            # Init 
            epochs_validation = 30
            num_samples = 200

            # HP and evaluate K-fold best config
            HP_and_valid_one_config(args,epochs_validation,vision_model_name,num_samples)
    if False:
        for dataset_names,vision_model_name in zip([['subway_in','netmob_POIs'],['netmob_POIs'],['subway_in']],['VariableSelectionNetwork','VariableSelectionNetwork',None]):
            args,_,_ = local_get_args(model_name,
                                    args_init = None,
                                    dataset_names=dataset_names,
                                    dataset_for_coverage=dataset_for_coverage,
                                    modification = {'ray':True,
                                                    'grace_period':2,
                                                    'HP_max_epochs':30,
                                                     'evaluate_complete_ds' : True})

            # Init 
            epochs_validation = 30
            num_samples = 200

            # HP and evaluate K-fold best config
            HP_and_valid_one_config(args,epochs_validation,vision_model_name,num_samples)