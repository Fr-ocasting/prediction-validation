

import os 
import sys
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Personnal import 

from examples.train_model_on_k_fold_validation import train_model_on_k_fold_validation

trial_id = 'subway_in_netmob_POIs_STGCN_VariableSelectionNetwork_MSELoss_2024_12_04_12_25_63821'
epochs_validation = 30
train_model_on_k_fold_validation(trial_id,load_config=True,save_folder='K_fold_validation/training_with_HP_tuning',epochs=epochs_validation,hp_tuning_on_first_fold = True)


if False:
    # GET PARAMETERS
    import sys
    import os
    import pandas as pd

    # Get Parent folder : 
    current_path = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_path, '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        
    from examples.benchmark import local_get_args,get_inputs,train_on_ds,get_trial_id

    def get_ds(model_name,dataset_names,dataset_for_coverage,vision_model_name = None):
        save_folder = None
        df_loss,df_results = pd.DataFrame(),pd.DataFrame()
        modification = {'epochs' : 10, #100,
                        }

        # Tricky but here we net to set 'netmob' so that we will use the same period for every combination
        args,folds,hp_tuning_on_first_fold = local_get_args(model_name,
                                                                dataset_names=dataset_names,
                                                                dataset_for_coverage=dataset_for_coverage,
                                                                modification = modification)
        trial_id = get_trial_id(args)
        K_fold_splitter,K_subway_ds,dic_class2rpz = get_inputs(args,vision_model_name,folds)
        ds = K_subway_ds[0]
        return(ds,args,trial_id,save_folder,dic_class2rpz,df_loss)

    dataset_names = ["netmob_image_per_station"] # ["subway_in","calendar"] # ["subway_in"] # ['data_bidon'] # ['METR_LA'] # ['PEMS_BAY']
    dataset_for_coverage = ['netmob_image_per_station'] #  ['data_bidon','netmob'] #  ['subway_in','netmob']  # ['METR_LA'] # ['PEMS_BAY']
    model_name = 'STGCN'
    vision_model_name =  'ImageAvgPooling'

    ds,args,trial_id,save_folder,dic_class2rpz,df_loss = get_ds(model_name,dataset_names,dataset_for_coverage,vision_model_name=vision_model_name)
    trainer,df_loss = train_on_ds(model_name,ds,args,trial_id,save_folder,dic_class2rpz,df_loss)
    Preds,Y_true,T_labels = trainer.testing(ds.normalizer, training_mode = 'test')