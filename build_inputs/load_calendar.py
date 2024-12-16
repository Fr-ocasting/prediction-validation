# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
#parent_dir = f"{parent_dir}/prediction_validation/"  # A utiliser sur .ipynb notebook
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

import argparse 
import  importlib
from dataset import TrainValidTest_Split_Normalize
from calendar_class import calendar_inputs,one_hot_encode_dataframe,get_time_slots_labels
from constants.paths import CALENDAR_TYPE


def load_calendar(subway_ds):
    """
    Load Calendar-input for the model. The choices of inputs is design with 'CALENDAR_TYPE'

    For each calendar related information ('dayofweek', 'hour', ...)
        get the tensor of sequences one_hot_dict[calendar_type] (size: [T,OHE-size])
        split into future feature vector U_train, U_valid, U_test
        keep these information into 'dict_calendar_U_{training_mode}'
    """
    dates = subway_ds.df_verif[f"t+{subway_ds.step_ahead-1}"]
    df_calendar = calendar_inputs(dates,city = subway_ds.city)
    one_hot_dict = one_hot_encode_dataframe(df_calendar, CALENDAR_TYPE)


    dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test = {},{},{}
    tensor_limits_keeper = subway_ds.tensor_limits_keeper
    for calendar_type in one_hot_dict.keys():
        calendar_tensor = one_hot_dict[calendar_type]
        splitter = TrainValidTest_Split_Normalize(calendar_tensor,
                            first_train = tensor_limits_keeper.first_train_U, last_train= tensor_limits_keeper.last_train_U,
                            first_valid= tensor_limits_keeper.first_valid_U, last_valid = tensor_limits_keeper.last_valid_U,
                            first_test = tensor_limits_keeper.first_test_U, last_test = tensor_limits_keeper.last_test_U)
        
        train_tensor_ds,valid_tensor_ds,test_tensor_ds = splitter.split_normalize_tensor_datasets(normalizer = None)

        dict_calendar_U_train[calendar_type] = train_tensor_ds.tensor
        dict_calendar_U_valid[calendar_type] =  valid_tensor_ds.tensor if valid_tensor_ds is not None else None
        dict_calendar_U_test[calendar_type] = test_tensor_ds.tensor if test_tensor_ds is not None else None

    return(dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test)

def get_args_embedding(args):
    if 'calendar' in args.dataset_names:
        module_path = f"dl_models.TimeEmbedding.load_config"
        module = importlib.import_module(module_path)
        importlib.reload(module)
        args_embedding = module.args
    else:
        args_embedding = argparse.ArgumentParser(description='TimeEmbedding').parse_args(args=[])

    args.args_embedding = args_embedding

    return(args)

"""
def load_calendar(subway_ds):
    '''Tackling Calendar Data''' 
    Dic_T_labels,Dic_class2rpz,Dic_rpz2class,Dic_nb_words_embedding = get_time_slots_labels(subway_ds)
    tensor_limits_keeper = subway_ds.tensor_limits_keeper

    dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test = {},{},{}
    for calendar_class in [0,1,2,3]:
        calendar_tensor = Dic_T_labels[calendar_class] #args.calendar_class

        splitter = TrainValidTest_Split_Normalize(calendar_tensor,
                                    first_train = tensor_limits_keeper.first_train_U, last_train= tensor_limits_keeper.last_train_U,
                                    first_valid= tensor_limits_keeper.first_valid_U, last_valid = tensor_limits_keeper.last_valid_U,
                                    first_test = tensor_limits_keeper.first_test_U, last_test = tensor_limits_keeper.last_test_U)

        train_tensor_ds,valid_tensor_ds,test_tensor_ds = splitter.split_normalize_tensor_datasets(normalizer = None)

        dict_calendar_U_train[calendar_class] = train_tensor_ds.tensor
        dict_calendar_U_valid[calendar_class] =  valid_tensor_ds.tensor if valid_tensor_ds is not None else None
        dict_calendar_U_test[calendar_class] = test_tensor_ds.tensor if test_tensor_ds is not None else None
    return(dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test,Dic_class2rpz,Dic_rpz2class,Dic_nb_words_embedding)

def get_args_embedding(args,nb_words_embedding,dic_class2rpz):
    if 'calendar' in args.dataset_names:
        module_path = f"dl_models.TimeEmbedding.load_config"
        module = importlib.import_module(module_path)
        importlib.reload(module)
        args_embedding = module.args
        args_embedding.nb_words_embedding = nb_words_embedding
        args_embedding.dic_class2rpz = dic_class2rpz
    else:
        args_embedding = argparse.ArgumentParser(description='TimeEmbedding').parse_args(args=[])

    args.args_embedding = args_embedding

    return(args)


def tackle_calendar(args,dic_class2rpz,dic_rpz2class,nb_words_embedding):
    if not('calendar' in args.dataset_names):
        dic_class2rpz,dic_rpz2class,nb_words_embedding = None,None,None
        
    args = get_args_embedding(args,nb_words_embedding,dic_class2rpz)

    args.dic_class2rpz = dic_class2rpz
    args.dic_rpz2class = dic_rpz2class
    args.nb_words_embedding = nb_words_embedding


    return args
"""