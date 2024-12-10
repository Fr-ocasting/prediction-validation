# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal inputs:
from build_inputs.load_netmob_data import tackle_netmob
from build_inputs.load_datasets_to_predict import load_datasets_to_predict
from build_inputs.load_calendar import load_calendar,tackle_calendar
from constants.paths import DATA_TO_PREDICT


def add_contextual_data(args,subway_ds,NetMob_ds,dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test):
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
    contextual_dataset_names = [dataset_name for dataset_name in args.dataset_names if dataset_name != DATA_TO_PREDICT]

    for dataset_name in contextual_dataset_names:
        if dataset_name == 'calendar':
            pos_calendar = list(contextual_tensors.keys()).index(f'calendar_{args.calendar_class}')
            positions['calendar'] = pos_calendar
            
        elif (dataset_name == 'netmob_image_per_station') or (dataset_name == 'netmob_bidon') or (dataset_name == 'netmob_video_lyon'):
             contextual_tensors.update({'netmob': {'train': NetMob_ds.U_train,
                                            'valid': NetMob_ds.U_valid if hasattr(NetMob_ds,'U_valid') else None,
                                            'test': NetMob_ds.U_test  if hasattr(NetMob_ds,'U_test') else None}
                                            }
                                            )
            
             pos_netmob = list(contextual_tensors.keys()).index('netmob')
             positions[dataset_name] = pos_netmob

        elif dataset_name == 'netmob_POIs':
             contextual_tensors.update({f'netmob_{NetMob_POI.station_name}': {'train': NetMob_POI.U_train,
                                            'valid': NetMob_POI.U_valid if hasattr(NetMob_POI,'U_valid') else None,
                                            'test': NetMob_POI.U_test  if hasattr(NetMob_POI,'U_test') else None}
                                            for NetMob_POI in NetMob_ds
                                            }
                                         )
             pos_netmob = [list(contextual_tensors.keys()).index(f'netmob_{NetMob_POI.station_name}') for NetMob_POI in NetMob_ds]

             positions[dataset_name] = pos_netmob

        else:
            raise NotImplementedError(f'Dataset {dataset_name} has not been implemented')


    subway_ds.contextual_tensors = contextual_tensors
    subway_ds.get_dataloader()

    # Maybe useless to send it to the both 
    subway_ds.contextual_positions = positions
    args.contextual_positions = positions

    return(subway_ds,args)



def load_complete_ds(args,coverage_period = None,vision_model_name = None,normalize = True):
    # Load subway-in DataSet:
    subway_ds,dataset,invalid_dates,intesect_coverage_period = load_datasets_to_predict(args,coverage_period,normalize)

    # Calendar data for Calibration : 
    dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test,dic_class2rpz,dic_rpz2class,nb_words_embedding = load_calendar(subway_ds)

    # Calendar data for training (with Time-Embedding):
    args,dic_class2rpz,dic_rpz2class,nb_words_embedding = tackle_calendar(args,dic_class2rpz,dic_rpz2class,nb_words_embedding)
    # Netmob: 
    args,NetMob_ds = tackle_netmob(dataset,invalid_dates,intesect_coverage_period,args,vision_model_name,normalize = normalize)
    # Add Contextual Tensors and their positions: 
    subway_ds,args = add_contextual_data(args,subway_ds,NetMob_ds,dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test)

    # Update/Set arguments: 
    assert subway_ds.U_train.dim() == 3, f'Feature Vector does not have the good dimension. Expected shape dimension [B,N,L], got {subway_ds.U_train.dim()} dim: {subway_ds.U_train.size()}'
    return(subway_ds,NetMob_ds,args,dic_class2rpz)