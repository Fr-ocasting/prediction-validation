from utils.utilities_DL import get_DataSet_and_invalid_dates
from build_inputs.preprocess_subway_15 import get_trigram_correspondance
from dataset import PersonnalInput

def load_subway_in(file_name,args,coverage):
    '''Tackling Subway_in data'''
    dataset,invalid_dates = get_DataSet_and_invalid_dates(args.abs_path, 'data/',file_name,
                                                        args.W,args.D,args.H,args.step_ahead,
                                                        single_station = False,coverage_period = coverage)

    # Change complete name to TRI-GRAM ( Ampère Victor Hugo -> AMP)
    df_correspondance = get_trigram_correspondance()
    df_correspondance.set_index('Station').reindex(dataset.columns)
    
    
    subway_ds = PersonnalInput(invalid_dates,args, tensor = dataset.raw_values, dates = dataset.df_dates,
                            time_step_per_hour = dataset.time_step_per_hour,Weeks = args.W, Days = args.D, historical_len = args.H,step_ahead = args.step_ahead,minmaxnorm = True ,dims=[0])
    

    # Set TRI-GRAM station
    subway_ds.columns = df_correspondance.COD_TRG
    subway_ds.preprocess(args.train_prop,args.valid_prop,args.test_prop)
    return(subway_ds,dataset,invalid_dates)
