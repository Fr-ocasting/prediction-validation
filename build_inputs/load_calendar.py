import torch 
# Relative path:
import sys 
import os 
import pandas as pd 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
#parent_dir = f"{parent_dir}/prediction_validation/"  # A utiliser sur .ipynb notebook
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

import argparse 
import  importlib
from dataset import TrainValidTest_Split_Normalize
from calendar_class import calendar_inputs,one_hot_encode_dataframe

def load_calendar(subway_ds):
    """
    Load Calendar-input for the model. The choices of inputs is design with 'CALENDAR_TYPE'

    For each calendar related information ('dayofweek', 'hour', ...)
        get the tensor of sequences one_hot_dict[calendar_type] (size: [T,OHE-size])
        split into future feature vector U_train, U_valid, U_test
        keep these information into 'dict_calendar_U_{training_mode}'
    """
    # Get date associated to last step ahead of prediction:
    dates = subway_ds.df_verif[f"t+{subway_ds.step_ahead-1}"]
    df_calendar = calendar_inputs(dates,city = subway_ds.city)

    dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test = {},{},{}
    tensor_limits_keeper = subway_ds.tensor_limits_keeper

    if 'calendar_embedding' in subway_ds.args.dataset_names:
        embedding_calendar_types = subway_ds.args.embedding_calendar_types
        # Can be in: ['dayofweek', 'hour', 'minute', 'bank_holidays', 'school_holidays', 'remaining_holidays']
        one_hot_dict = one_hot_encode_dataframe(df_calendar, embedding_calendar_types)  # Get one-hot encoded vector for each specific calendar type 
        for calendar_type in one_hot_dict.keys():
            calendar_tensor = one_hot_dict[calendar_type]
            dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test = update_dict_calendar(dict_calendar_U_train,
                                                                                            dict_calendar_U_valid,
                                                                                            dict_calendar_U_test,
                                                                                            calendar_tensor,
                                                                                            tensor_limits_keeper,
                                                                                            f"{calendar_type}_OHE")
            
    if 'calendar' in subway_ds.args.dataset_names:
        calendar_types = subway_ds.args.calendar_types
        # Get dates associated to 
        date_related_tensors = {calendar_type: pd.DataFrame() for calendar_type in calendar_types}
        for c in subway_ds.df_verif.columns[:-subway_ds.step_ahead]:
            date_i = subway_ds.df_verif[c]
            df_calendar_i = calendar_inputs(dates,city = subway_ds.city)
            for calendar_type in calendar_types:
                date_related_tensors[calendar_type][c] = df_calendar_i[calendar_type]
                
        # ...

        for calendar_type in calendar_types:
            calendar_tensor = torch.tensor(date_related_tensors[calendar_type].values, dtype=torch.float32)
            if calendar_tensor.dim() == 1: calendar_tensor = calendar_tensor.unsqueeze(1)
            dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test = update_dict_calendar(dict_calendar_U_train,
                                                                                                dict_calendar_U_valid,
                                                                                                dict_calendar_U_test,
                                                                                                calendar_tensor,
                                                                                                tensor_limits_keeper,
                                                                                                calendar_type)

    return(dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test)

def update_dict_calendar(dict_calendar_U_train, dict_calendar_U_valid,dict_calendar_U_test,
                          calendar_tensor,tensor_limits_keeper, calendar_type):

    splitter = TrainValidTest_Split_Normalize(calendar_tensor,
                        first_train = tensor_limits_keeper.first_train_U, last_train= tensor_limits_keeper.last_train_U,
                        first_valid= tensor_limits_keeper.first_valid_U, last_valid = tensor_limits_keeper.last_valid_U,
                        first_test = tensor_limits_keeper.first_test_U, last_test = tensor_limits_keeper.last_test_U)
    
    train_tensor_ds,valid_tensor_ds,test_tensor_ds = splitter.split_normalize_tensor_datasets(normalizer = None)

    dict_calendar_U_train[calendar_type] = train_tensor_ds.tensor
    dict_calendar_U_valid[calendar_type] =  valid_tensor_ds.tensor if valid_tensor_ds is not None else None
    dict_calendar_U_test[calendar_type] = test_tensor_ds.tensor if test_tensor_ds is not None else None

    return dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test 


def dim_by_calendar_type(calendar_type):
    calendar_type = calendar_type.split('_OHE')[0]
    #['dayofweek', 'hour', 'minute', 'bank_holidays', 'school_holidays', 'remaining_holidays']
    if calendar_type == 'dayofweek':
        return(3)
    elif calendar_type == 'hour':
        return(5)
    elif calendar_type == 'minute':
        return(2)
    elif calendar_type == 'bank_holidays':
        return(1)
    elif calendar_type == 'school_holidays':
        return(1)
    elif calendar_type == 'remaining_holidays':
        return(1)
    else:
        raise NotImplementedError(f"Unknown calendar type: {calendar_type}. Please check the calendar_types list here")
    
def get_args_embedding(args,dict_calendar_U_train):
    if 'calendar_embedding' in args.dataset_names:
        module_path = f"dl_models.TimeEmbedding.load_config"
        module = importlib.import_module(module_path)
        importlib.reload(module)
        args_embedding = module.args
        print(dict_calendar_U_train.keys())
        print('args_embedding.variable_selection_model_name: ',args_embedding.variable_selection_model_name)
        args_embedding.dic_sizes = [max(2,dict_calendar_U_train[calendar_type].size(-1)) for calendar_type in dict_calendar_U_train.keys()]
        args_embedding.embedding_dim_calendar_units = [max(1,dim_by_calendar_type(calendar_type)) for calendar_type in dict_calendar_U_train.keys()]
        args_embedding.device = args.device
    else:
        args_embedding = argparse.ArgumentParser(description='TimeEmbedding').parse_args(args=[])

    args.args_embedding = args_embedding

    return(args)