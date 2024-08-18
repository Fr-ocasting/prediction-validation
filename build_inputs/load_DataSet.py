import pandas as pd
from datetime import datetime

# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal inputs:
from DL_class import DataSet


# ======================================================
# Function 
# ======================================================

# Load DataSet object and Sequences : 
def load_normalized_dataset(df,time_step_per_hour,train_prop,valid_prop,step_ahead,H,D,W,invalid_dates=[]):

    # Load DataSet object
    dataset = DataSet(df,time_step_per_hour=time_step_per_hour, Weeks = W, Days = D, historical_len= H,step_ahead=step_ahead)
    dataset.remove_invalid_dates(invalid_dates)  # remove invalid_dates. Build 'remaining_dataset'
    dataset.get_feature_vect()  # Build 'df_verif'. Length of df_verif = number of sequences 
    dataset.train_valid_test_limits(train_prop,valid_prop)
    # MinMax Normalize, without taking into account the invalid_dates
    dataset.normalize_df(invalid_dates,minmaxnorm=True)

    # Built Feature Vector 

    # ========= AVANT ON AVAIT NORMALIZE PUIS GET_FEAUTRE_VECT =======
    #(U,Utarget,df_verif) = dataset.get_feature_vect()
    # =====================================================

    # Identify Invalid index on the feature vector 
    invalid_indices_tensor,invalid_indx_df = dataset.get_invalid_indx(invalid_dates)  # has to be run after 'get_feature_vect'

    # Remove the Invalid Sequences 
    Uclean,Utarget_clean,remaining_dates = dataset.remove_indices(invalid_indices_tensor)

    return(dataset,Uclean,Utarget_clean,remaining_dates)


# ======================================================
# Application 
# ======================================================
if __name__ == '__main__':

    # Init
    folder_path = 'data/'
    file_name = 'preprocessed_subway_15_min.csv'

    subway_in = pd.read_csv(folder_path+file_name,index_col = 0)
    subway_in.columns.name = 'Station'
    subway_in.index = pd.to_datetime(subway_in.index)

    time_step_per_hour=4
    train_prop = 0.6

    # Set forbidden dates :
    # Data from  23_03_2019 14:00:00 to 28_04_2019 12:00:00 included should not been taken into account 
    invalid_dates = pd.date_range(datetime(2019,4,23,14),datetime(2019,4,28,14),freq = f'{60/time_step_per_hour}min')

    (dataset_in,U_in,Utarget_in) = load_normalized_dataset(subway_in,time_step_per_hour,train_prop,invalid_dates)

    # colname2indx allow to keep track on the position of a station ('Ampere', 'Brotteaux', ...) within the Tensor
    colname2indx_in,indx2colname_in = dataset_in.bijection_name_indx()


