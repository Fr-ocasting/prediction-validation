# Relative path:
import sys 
import os 
import torch
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal inputs:
from utils.utilities_DL import get_DataSet_and_invalid_dates
from build_inputs.preprocess_subway_15 import get_trigram_correspondance
from dataset import PersonnalInput

def preprocess_subway_in(dataset,args,invalid_dates,normalize = True):
    # Change complete name to TRI-GRAM ( Ampère Victor Hugo -> AMP)
    df_correspondance = get_trigram_correspondance()
    df_correspondance.set_index('Station').reindex(dataset.columns)
    
    print('\nInit Subway-In Dataset: ', dataset.raw_values.size())
    print('Number of Nan Value: ',torch.isnan(dataset.raw_values).sum())
    print('Total Number of Elements: ', dataset.raw_values.numel(),'\n')

    subway_ds = PersonnalInput(invalid_dates,args, tensor = dataset.raw_values, dates = dataset.df_dates,
                            time_step_per_hour = dataset.time_step_per_hour,Weeks = args.W, Days = args.D, historical_len = args.H,step_ahead = args.step_ahead,minmaxnorm = True ,dims=[0])


    # Set TRI-GRAM station
    subway_ds.columns = df_correspondance.COD_TRG
    subway_ds.preprocess(args.train_prop,args.valid_prop,args.test_prop,normalize)
    
    return(subway_ds)

def load_subway_in(file_name,args,coverage,normalize=True,time_slot_limits = None):
    '''Tackling Subway_in data
    
    outputs:
    --------
    subway_ds : PersonnalInput object, containing dataset.raw_values
    invalid_dates : All the dates which have been removed 

    '''
    dataset,invalid_dates = get_DataSet_and_invalid_dates(args.abs_path, 'data/',file_name,
                                                        args.W,args.D,args.H,args.step_ahead,args.dataset_names,
                                                        single_station = False,coverage_period = coverage)
    dataset.time_slot_limits = time_slot_limits
    
    subway_ds = preprocess_subway_in(dataset,args,invalid_dates,normalize)

    
    return(subway_ds,dataset,invalid_dates)
