from dataset import TrainValidTest_Split_Normalize
from calendar_class import get_time_slots_labels
from utils.utilities_DL import get_args_embedding

def load_calendar(subway_ds):
    '''Tackling Calendar Data''' 
    time_slots_labels,dic_class2rpz,dic_rpz2class,nb_words_embedding = get_time_slots_labels(subway_ds)
    tensor_limits_keeper = subway_ds.tensor_limits_keeper

    dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test = {},{},{}
    for calendar_class in [0,1,2,3]:
        calendar_tensor = time_slots_labels[calendar_class] #args.calendar_class

        splitter = TrainValidTest_Split_Normalize(calendar_tensor,
                                    first_train = tensor_limits_keeper.first_train_U, last_train= tensor_limits_keeper.last_train_U,
                                    first_valid= tensor_limits_keeper.first_valid_U, last_valid = tensor_limits_keeper.last_valid_U,
                                    first_test = tensor_limits_keeper.first_test_U, last_test = tensor_limits_keeper.last_test_U)

        train_tensor_ds,valid_tensor_ds,test_tensor_ds = splitter.split_normalize_tensor_datasets(normalizer = None)
        calendar_U_train,calendar_U_valid,calendar_U_test = train_tensor_ds.tensor,valid_tensor_ds.tensor,test_tensor_ds.tensor
        dict_calendar_U_train[calendar_class] = calendar_U_train
        dict_calendar_U_valid[calendar_class] = calendar_U_valid
        dict_calendar_U_test[calendar_class] = calendar_U_test
    return(dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test,dic_class2rpz,dic_rpz2class,nb_words_embedding)



def tackle_calendar(dataset_names,args,dic_class2rpz,dic_rpz2class,nb_words_embedding):
    if 'calendar' in dataset_names:
        args.time_embedding = True
    else:
        dic_class2rpz,dic_rpz2class,nb_words_embedding = None,None,None
        args.time_embedding = False
        
    args_embedding = get_args_embedding(args,nb_words_embedding)

    return args,dic_class2rpz,dic_rpz2class,nb_words_embedding,args_embedding