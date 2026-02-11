import sys
import os
import gc
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np 
from constants.config import get_args,update_modif, modification_contextual_args


from pipeline.utils.save_results import get_trial_id

import matplotlib.pyplot as plt 
import importlib







def keep_track_on_model_metrics(trainer,df_results,model_name,performance,metrics):
    performance = trainer.performance
    dict_row = {'Model':[model_name],
                'Valid_loss':[performance['valid_loss']]
                }
    for metric in metrics:
        if (metric == 'PICP') or (metric == 'MPIW'):
            add_name = f'calib_{trainer.type_calib}_'
        else:
            add_name = ''
        dict_row.update({f'Valid_{add_name}{metric}':[performance['valid_metrics'][metric]],
                        f'Test_{add_name}{metric}':[performance['test_metrics'][metric]]
                        })

    row = pd.DataFrame(dict_row)
    df_results = pd.concat([df_results,row])
    return df_results






























if __name__ == '__main__':

    def plot_gradient(trainer):
        gradient_metrics = trainer.gradient_metrics
        for module in gradient_metrics.keys():
            df_to_plot = pd.DataFrame()
            for metric_grad in gradient_metrics[module].keys():
                df_to_plot[f"{module}_{metric_grad}"] = gradient_metrics[module][metric_grad]
            if len(df_to_plot) > 0: 
                if len(df_to_plot)> 3:
                    df_to_plot.iloc[3:].plot()
                else:
                    df_to_plot.plot()
    for dataset_names,vision_model_name in zip([['subway_in','calendar']],[None]): # zip([['subway_in','netmob_POIs','calendar'],['subway_in']],['VariableSelectionNetwork',None]):
        # GET PARAMETERS
        #dataset_names = ['subway_in','netmob_POIs'] # ["subway_in","calendar"] # ["subway_in"] # ['data_bidon'] # ['METR_LA'] # ['PEMS_BAY']  # ['data_bidon','netmob_bidon'] #['netmob_POIs']
        dataset_for_coverage = ['subway_in','netmob_POIs'] #['subway_in','netmob_POIs'] #  ['data_bidon','netmob'] #  ['subway_in','netmob']  # ['METR_LA'] # ['PEMS_BAY'] # ['data_bidon','netmob_bidon'] #['netmob_POIs'] 
        #vision_model_name = 'VariableSelectionNetwork' # None # 'VariableSelectionNetwork'


        save_folder = 'benchmark/fold0/'
        df_loss,df_results = pd.DataFrame(),pd.DataFrame()
        modification = {'epochs' : 2, #100,
                        'lr':4e-4,
                        'set_spatial_units' : ['BON','SOI','GER','CHA'],
                        'TE_concatenation_early':True,
                        'TE_concatenation_late':False,
                        'TE_embedding_dim':4,
                        'TE_variable_selection_model_name': 'GRN'
                        }
    
        model_names =['STGCN'] # [None] # ['CNN','LSTM','GRU','RNN','STGCN'] #'DCRNN','MTGNN'
        print(f'\n>>>>Training {model_names[0]} on {dataset_names}')
        # Tricky but here we net to set 'netmob' so that we will use the same period for every combination
        args = local_get_args(model_names[0],
                                args_init = None,
                                dataset_names=dataset_names,
                                dataset_for_coverage=dataset_for_coverage,
                                modification = modification)
        K_fold_splitter,K_subway_ds,args = get_inputs(args,folds=[0])
        args = modification_contextual_args(args,modification)
        trial_id = get_trial_id(args)
        ds = K_subway_ds[0]

        trainer,df_loss = train_on_ds(ds,args,trial_id,save_folder,df_loss)
        metrics = trainer.metrics
        df_results = keep_track_on_model_metrics(trainer,df_results,model_names[0],trainer,metrics)

        # Display Gradient : 
        plot_gradient(trainer)

        for model_name in model_names[1:]:  # benchamrk on all the other models, with the same input base['MTGNN','STGCN', 'CNN', 'DCRNN']
            print(f'\n>>>>Training {model_name} on {dataset_names}')
            args= local_get_args(model_name,
                                args_init = args,
                                dataset_names=dataset_names,
                                dataset_for_coverage=dataset_for_coverage,
                                modification = modification)
            trial_id = get_trial_id(args)
            trainer,df_loss = train_on_ds(ds,args,trial_id,save_folder,df_loss)
            metrics = trainer.metrics
            df_results = keep_track_on_model_metrics(trainer,df_results,model_name,trainer,metrics)

        print(df_results)
        df_loss[[f"{model}_valid_loss" for model in model_names]].plot()
        plt.show()
        df_results.to_csv(f'{parent_dir}/save/results/{trial_id}.csv')

        del args 
        del ds
        del K_fold_splitter
        del K_subway_ds
        del trainer 
        del df_results
        gc.collect()

       

