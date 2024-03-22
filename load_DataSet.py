import pandas as pd
import glob
from datetime import datetime

import numpy as np 

# Personnal Import 
# from DL_utilities import DataSet
from DL_class import DataSet
# ======================================================
# Function 
# ======================================================

# Load DataSet object and Sequences : 
def load_normalized_dataset(df,time_step_per_hour,train_prop,step_ahead,H,D,W,invalid_dates=[]):

    # Load DataSet object
    dataset = DataSet(df,time_step_per_hour=time_step_per_hour)

    # MinMax Normalize, without taking into account the invalid_dates
    dataset.normalize_df(train_prop,invalid_dates)

    # Built Feature Vector 
    (U,Utarget,df_verif) = dataset.get_feature_vect(step_ahead,H,D,W)

    # Identify Invalid index on the feature vector 
    invalid_indices_tensor,invalid_indx_df = dataset.get_invalid_indx(invalid_dates,df_verif)  # has to be run after 'get_feature_vect'

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


