# Relative path:
import sys 
import os 
import torch
import importlib
import pandas as pd
current_file_path = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(current_file_path,'..'))
if ROOT not in sys.path:
    sys.path.insert(0,ROOT)
# ...

# Personnal inputs:
from dataset import PersonnalInput,DataSet
from constants.paths import FOLDER_PATH
from utils.utilities import filter_args,get_INVALID_DATES
from utils.seasonal_decomposition import fill_and_decompose_df

"""
def add_noise(preprocesed_ds,args):
    if args.data_augmentation and args.DA_method == 'noise':
        if args.DA_noise_from == 'MSTL':
            decomposition = fill_and_decompose_df(preprocesed_ds.raw_values,
                                                preprocesed_ds.tensor_limits_keeper.df_verif_train,
                                                preprocesed_ds.time_step_per_hour,
                                                preprocesed_ds.spatial_unit,
                                                min_count = args.DA_min_count, 
                                                periods = preprocesed_ds.periods,
                                                dataset_name =dataset_name)
            df_noises = pd.DataFrame({col : decomposition[col]['resid'] for col in decomposition.keys()})
            df_noises = df_noises[preprocesed_ds.spatial_unit]
        elif args.DA_noise_from == 'Homogenous':
            df_verif_train = preprocesed_ds.tensor_limits_keeper.df_verif_train
            dates_used_in_train = pd.Series(pd.concat([df_verif_train[c] for c in df_verif_train.columns]).unique()).sort_values() 
            reindex_dates = pd.date_range(dates_used_in_train.min(),dates_used_in_train.max(),freq=f"{1/preprocesed_ds.time_step_per_hour}h")
            reindex_dates = reindex_dates[~reindex_dates.hour.isin(USELESS_DATES['hour'])&~reindex_dates.hour.isin(USELESS_DATES['weekday'])]
            df_noises = pd.DataFrame({col : [1]*len(reindex_dates) for col in preprocesed_ds.spatial_unit},index =reindex_dates)
        else :
            raise NotImplementedError(f"Noise from {args.DA_noise_from} has not been implemented")
        preprocesed_ds.noises = {preprocesed_ds.target_data:df_noises}
    else:
        preprocesed_ds.noises = {}
    return preprocesed_ds
"""

def get_intersect_of_coverage_periods(args,coverage_period):
     # Load the Intersection of all the coverage period of each dataset_name:
    list_of_list_coverage_period,list_of_list_invalid_dates = [],[]
    for ds_name in list(set(args.dataset_for_coverage)|set(args.dataset_names)):
        data_module = importlib.import_module(f"load_inputs.{ds_name}")
        importlib.reload(data_module) 
        
        coverage_i = pd.date_range(start=data_module.START, end=data_module.END, freq=args.freq)[:-1]
        list_of_list_coverage_period.append(coverage_i)

        invalid_dates_i = get_INVALID_DATES(data_module.list_of_invalid_period,args.freq)
        list_of_list_invalid_dates.append(invalid_dates_i)

    intersect_coverage_period = sorted(list(set.intersection(*map(set, list_of_list_coverage_period))))
    # ___Intersection between the expected coverage_period and the limits from datasets:
    if coverage_period is not None: 
        intersect_coverage_period = sorted(list(set(coverage_period)&set(intersect_coverage_period)))
    # ...
       
    # Load the union of all the invalid_dates: 
    union_invalid_dates = list(set.union(*map(set, list_of_list_invalid_dates)))
    # ___Restrain the invalid dates to the specific restained coverage period :
    union_invalid_dates = sorted(list(set(union_invalid_dates)&set(intersect_coverage_period)))
    if len(intersect_coverage_period)==0:
        raise ValueError("Intersection of coverage period is empty. Check the coverage period of each dataset and the expected coverage period\n\
                         Coverage period of each dataset:\n",'\n'.join([f'{importlib.import_module(f"load_inputs.{ds_name}").START} - {importlib.import_module(f"load_inputs.{ds_name}").END}' for ds_name in list(set(args.dataset_for_coverage)|set(args.dataset_names))])
        )
    print(f"Coverage Period: {len(intersect_coverage_period)} elts between {min(intersect_coverage_period)} and {max(intersect_coverage_period)}") 
    print('Invalid dates within this fold:',len(union_invalid_dates))
    return union_invalid_dates,intersect_coverage_period



def load_datasets_to_predict(args,invalid_dates,coverage_period,normalize=True,):
    '''Tackling DataSet to predict : Subway_in data,
    
    outputs:
    --------
    subway_ds : PersonnalInput object, containing dataset.raw_values
    invalid_dates : All the dates which have been removed 
    '''
    # Load the dataset and its associated caracteristics
    print('\n>>>Tackle Target dataset:',args.target_data)
    module_data = importlib.import_module(f"load_inputs.{args.target_data}")
    importlib.reload(module_data) 
    
    preprocessed_ds = module_data.load_data(ROOT,FOLDER_PATH,
                                            invalid_dates = invalid_dates,
                                            coverage_period = coverage_period,
                                            args = args,
                                            normalize= True)
    print(f"   Init Dataset: '{preprocessed_ds.raw_values.size()}. {torch.isnan(preprocessed_ds.raw_values).sum()} Nan values")
    print('   TRAIN contextual_ds:',preprocessed_ds.U_train.size())
    print('   VALID contextual_ds:',preprocessed_ds.U_valid.size()) if hasattr(preprocessed_ds,'U_valid') else None
    print('   TEST contextual_ds:',preprocessed_ds.U_test.size()) if hasattr(preprocessed_ds,'U_test') else None

    if args.data_augmentation and args.DA_method == 'noise':
        if args.DA_noise_from == 'MSTL':
            raise NotImplementedError("Has to decompose seasonal component first but not implemented yet. Please refer to the commented code in the top of this file.")
    args.n_vertex = preprocessed_ds.n_vertex
    args.C = preprocessed_ds.C
    # ...
  
    return(preprocessed_ds)
