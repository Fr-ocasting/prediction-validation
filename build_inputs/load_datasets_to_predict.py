# Relative path:
import sys 
import os 
import torch
import importlib
import pandas as pd
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal inputs:
from dataset import PersonnalInput,DataSet
from constants.paths import DATA_TO_PREDICT,FOLDER_PATH,USELESS_DATES
from utils.utilities import filter_args,get_INVALID_DATES
from utils.seasonal_decomposition import fill_and_decompose_df

def preprocess_dataset(dataset,args,invalid_dates,normalize = True): 
    print(f"\nInit Dataset: '{dataset.raw_values.size()} with {dataset.raw_values.numel()} Total nb of elements and {torch.isnan(dataset.raw_values).sum()} Nan values")

    args_DataSet = filter_args(DataSet, args)
    print(f'nb {DATA_TO_PREDICT} invalid dates: ',len(invalid_dates))
    preprocesed_ds = PersonnalInput(invalid_dates,args,
                                     tensor = dataset.raw_values, 
                                     dates = dataset.df_dates, 
                                     spatial_unit = dataset.spatial_unit,
                                     indices_spatial_unit = dataset.indices_spatial_unit,
                                     time_step_per_hour = dataset.time_step_per_hour,
                                     minmaxnorm = dataset.minmaxnorm,
                                     standardize = dataset.standardize,
                                     city = dataset.city,
                                     dims=dataset.dims,
                                     periods = dataset.periods,
                                     **args_DataSet
                                     )

    preprocesed_ds.preprocess(args.train_prop,args.valid_prop,args.test_prop,args.train_valid_test_split_method,normalize)
    preprocesed_ds = add_noise(preprocesed_ds,args)

    return(preprocesed_ds)


def add_noise(preprocesed_ds,args):
    if args.data_augmentation and args.DA_method == 'noise':
        if args.DA_noise_from == 'MSTL':
            decomposition = fill_and_decompose_df(preprocesed_ds.raw_values,
                                                preprocesed_ds.tensor_limits_keeper.df_verif_train,
                                                preprocesed_ds.time_step_per_hour,
                                                preprocesed_ds.spatial_unit,
                                                min_count = args.DA_min_count, 
                                                periods = preprocesed_ds.periods)
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
        preprocesed_ds.noises = {DATA_TO_PREDICT:df_noises}
    else:
        preprocesed_ds.noises = {}
    return preprocesed_ds



def get_intersect_of_coverage_periods(args,coverage_period):
     # Load the Intersection of all the coverage period of each dataset_name:
    list_of_list_coverage_period,list_of_list_invalid_dates = [],[]
    for ds_name in args.dataset_for_coverage:
        data_module = importlib.import_module(f"load_inputs.{ds_name}")
        importlib.reload(data_module) 
        coverage_i = pd.date_range(start=data_module.START, end=data_module.END, freq=args.freq)[:-1]
        list_of_list_coverage_period.append(coverage_i)
        invalid_dates_i = get_INVALID_DATES(data_module.list_of_invalid_period,args.freq)

        list_of_list_invalid_dates.append(invalid_dates_i)

    intersect_coverage_period = list(set.intersection(*map(set, list_of_list_coverage_period)))
    # ___Intersection between the expected coverage_period and the limits from datasets:
    if coverage_period is not None: 
        intersect_coverage_period = list(set(coverage_period)&set(intersect_coverage_period))
    # ...
       
    # Load the union of all the invalid_dates: 
    union_invalid_dates = list(set.union(*map(set, list_of_list_invalid_dates)))
    # ___Restrain the invalid dates to the specific restained coverage period :
    union_invalid_dates = list(set(union_invalid_dates)&set(intersect_coverage_period))
    print('Invalid dates within this fold:',len(union_invalid_dates))
    return union_invalid_dates,intersect_coverage_period



def load_datasets_to_predict(args,coverage_period,normalize=True):
    '''Tackling DataSet to predict : Subway_in data,
    
    outputs:
    --------
    subway_ds : PersonnalInput object, containing dataset.raw_values
    invalid_dates : All the dates which have been removed 
    '''
    union_invalid_dates,intersect_coverage_period =get_intersect_of_coverage_periods(args,coverage_period)

    # Load the dataset and its associated caracteristics
    module_data = importlib.import_module(f"load_inputs.{DATA_TO_PREDICT}")
    importlib.reload(module_data) 
    dataset = module_data.load_data(args,parent_dir,FOLDER_PATH,intersect_coverage_period)
    args.n_vertex = dataset.n_vertex
    args.C = dataset.C

    # ...
    preprocesed_ds = preprocess_dataset(dataset,args,union_invalid_dates,normalize)
    return(preprocesed_ds,dataset,union_invalid_dates,intersect_coverage_period)
