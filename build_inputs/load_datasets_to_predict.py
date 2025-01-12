# Relative path:
import sys 
import os 
import torch
import importlib
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal inputs:
from dataset import PersonnalInput
from constants.paths import DATA_TO_PREDICT,FOLDER_PATH


def preprocess_dataset(dataset,args,invalid_dates,normalize = True): 
    print('\nInit Dataset: ', dataset.raw_values.size())
    print('Number of Nan Value: ',torch.isnan(dataset.raw_values).sum())
    print('Total Number of Elements: ', dataset.raw_values.numel(),'\n')

    preprocesed_ds = PersonnalInput(invalid_dates,args,
                                     tensor = dataset.raw_values, 
                                     dates = dataset.df_dates, 
                                     spatial_unit = dataset.spatial_unit,
                                    indices_spatial_unit = dataset.indices_spatial_unit,
                                     time_step_per_hour = dataset.time_step_per_hour,
                                     Weeks = args.W, 
                                     Days = args.D, 
                                     historical_len = args.H,
                                     step_ahead = args.step_ahead,
                                     minmaxnorm = True ,
                                     city = dataset.city,
                                     dims=dataset.dims)

    preprocesed_ds.preprocess(args.train_prop,args.valid_prop,args.test_prop,args.train_valid_test_split_method,normalize)
    
    return(preprocesed_ds)

def get_intersect_of_coverage_periods(args,coverage_period):
     # Load the Intersection of all the coverage period of each dataset_name:
    list_of_list_coverage_period,list_of_list_invalid_dates = [],[]
    for ds_name in args.dataset_for_coverage:
        data_module = importlib.import_module(f"load_inputs.{ds_name}")
        importlib.reload(data_module) 
        list_of_list_coverage_period.append(data_module.COVERAGE)
        list_of_list_invalid_dates.append(data_module.INVALID_DATES)

    intesect_coverage_period = list(set.intersection(*map(set, list_of_list_coverage_period)))
    # ___Intersection between the expected coverage_period 
    if coverage_period is not None: 
        intesect_coverage_period = list(set(coverage_period)&set(intesect_coverage_period))
    # ...
       
    # Load the union of all the invalid_dates: 
    union_invalid_dates = list(set.union(*map(set, list_of_list_invalid_dates)))
    # ___Restrain the invalid dates to the specific restained coverage period :
    union_invalid_dates = list(set(union_invalid_dates)&set(intesect_coverage_period))
    return union_invalid_dates,intesect_coverage_period



def load_datasets_to_predict(args,coverage_period,normalize=True):
    '''Tackling DataSet to predict : Subway_in data,
    
    outputs:
    --------
    subway_ds : PersonnalInput object, containing dataset.raw_values
    invalid_dates : All the dates which have been removed 
    '''
    union_invalid_dates,intesect_coverage_period =get_intersect_of_coverage_periods(args,coverage_period)

    # Load the dataset and its associated caracteristics
    module_data = importlib.import_module(f"load_inputs.{DATA_TO_PREDICT}")
    importlib.reload(module_data) 
    #args.n_vertex = module_data.n_vertex
    #args.C = module_data.C
    dataset = module_data.load_data(args,parent_dir,FOLDER_PATH,intesect_coverage_period)
    # ...
    preprocesed_ds = preprocess_dataset(dataset,args,union_invalid_dates,normalize)
    return(preprocesed_ds,dataset,union_invalid_dates,intesect_coverage_period)
