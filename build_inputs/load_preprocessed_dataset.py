
from build_inputs.load_netmob_data import tackle_netmob
from build_inputs.load_subway_in import load_subway_in
from build_inputs.load_calendar import load_calendar,tackle_calendar
from constants.config import update_args


def add_contextual_data(dataset_names,args,subway_ds,NetMob_ds,dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test):
    # === Define DataLoader : 
    contextual_tensors,positions = {},{}

    # Define contextual tensor for Calibration with Calendar Class:
    contextual_tensors = {f'calendar_{calendar_class}': {'train': dict_calendar_U_train[calendar_class],
                                'valid': dict_calendar_U_valid[calendar_class],
                                'test': dict_calendar_U_test[calendar_class]} for calendar_class in dict_calendar_U_train.keys()
                                } 
    # ...
    pos_calibration_calendar = list(contextual_tensors.keys()).index(f'calendar_{args.calibration_calendar_class}')
    positions['calibration_calendar'] = pos_calibration_calendar

    if 'calendar' in dataset_names:
        pos_calendar = list(contextual_tensors.keys()).index(f'calendar_{args.calendar_class}')
        positions['calendar'] = pos_calendar
        

    if 'netmob' in dataset_names:
        contextual_tensors.update({'netmob': {'train': NetMob_ds.U_train,
                                        'valid': NetMob_ds.U_valid if hasattr(NetMob_ds,'U_valid') else None,
                                        'test': NetMob_ds.U_test  if hasattr(NetMob_ds,'U_test') else None}
                                        }
                                        )
        
        pos_netmob = list(contextual_tensors.keys()).index('netmob')
        positions['netmob'] = pos_netmob



    subway_ds.contextual_tensors = contextual_tensors
    subway_ds.get_dataloader()
    subway_ds.contextual_positions = positions

    return(subway_ds)



def load_complete_ds(dataset_names,args,coverage,folder_path,file_name,vision_model_name, normalize = True):

    # Load subway-in DataSet:
    subway_ds,dataset,invalid_dates = load_subway_in(file_name,args,coverage,normalize)

    # Calendar data for Calibration : 
    dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test,dic_class2rpz,dic_rpz2class,nb_words_embedding = load_calendar(subway_ds)

    # Calendar data for training (with Time-Embedding):
    args,dic_class2rpz,dic_rpz2class,nb_words_embedding,args_embedding = tackle_calendar(dataset_names,args,dic_class2rpz,dic_rpz2class,nb_words_embedding)

    # Netmob: 
    args_vision,NetMob_ds = tackle_netmob(dataset,dataset_names,invalid_dates,args,folder_path,subway_ds.columns,vision_model_name,normalize = normalize)
    
    # Add Contextual Tensors and their positions: 
    subway_ds = add_contextual_data(dataset_names,args,subway_ds,NetMob_ds,dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test)

    # Update/Set arguments: 
    args = update_args(args,subway_ds,dataset_names)
    return(subway_ds,NetMob_ds,args,args_vision,args_embedding,dic_class2rpz)