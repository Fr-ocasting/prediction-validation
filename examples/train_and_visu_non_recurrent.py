# GET PARAMETERS
import sys
import os
import pandas as pd
import numpy as np 
import geopandas as gpd 

# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from examples.benchmark import local_get_args,get_inputs,train_on_ds
from utils.save_results import get_trial_id
from constants.config import modification_contextual_args
from plotting.TS_analysis import drag_selection_box,plot_single_point_prediction,plot_prediction_error,plot_loss_from_trainer,plot_TS
from bokeh.plotting import show,output_notebook
from bokeh.layouts import column,row
from utils.specific_event import rugby_matches
from constants.paths import FOLDER_PATH

RANGE = 3*60  # +/- range (min) supposed to be affected around the event 
WIDTH = 1000
HEIGHT = 300
MIN_FLOW = 20



def evaluate_config(model_name,dataset_names,dataset_for_coverage,transfer_modes= None,
                    type_POIs = None,
                    spatial_units = None,
                    apps = None,
                    POI_or_stations = None,
                    expanded =None,
                    modification = {},
                    station=['GER'],
                    training_mode_to_visualise = ['test','valid','train']
                    ):
    
    '''
    args: 
    type_POIs : list of type of POIs.                         >>> ['stadium','nightclub']
    spatial_units : list of name of spatial units to analyse. >>> ['Lou_rugby','Ninkasi_Kao']
    apps : list of apps to deal with.                         >>> ['Instagram']
    POI_or_stations : list of type of object to analyse contains occurence of 'POI' or 'station'.  >>> ['POI']
    expanded: '' if we look at the intensity of netmob consumption at the POI. '_expanded' if we look also one square around.
    '''
    ds,args,trial_id,save_folder,df_loss = get_ds(model_name,dataset_names,dataset_for_coverage,modification=modification)
    trainer,df_loss = train_on_ds(ds,args,trial_id,save_folder,df_loss)
    # Allow us to have 'dataloader['train'] with no shuffle !!!!
    # ======
    modification.update({'shuffle':False })
    ds_no_shuffle,args_no_shuffle,trial_id,save_folder,df_loss =  get_ds(model_name,dataset_names,
                                                                            dataset_for_coverage, 
                                                                            modification = modification)

    trainer.dataloader = ds_no_shuffle.dataloader
    # ======

    for training_mode in training_mode_to_visualise:
        analysis_on_specific_training_mode(trainer,ds_no_shuffle,
                                           training_mode=training_mode,
                                           transfer_modes= transfer_modes,
                                           type_POIs = type_POIs,
                                           spatial_units=spatial_units,
                                           apps=apps,
                                           POI_or_stations = POI_or_stations,
                                           expanded=expanded,
                                           station=station)
    return(trainer,ds,ds_no_shuffle,args)

def netmob_volume_on_POI(gdf_POI_2_tile_ids,app = 'Instagram',transfer_mode = 'DL',type_POI = 'stadium', spatial_unit = 'Lou_rugby',POI_or_station='POI',expanded=''):

    gdf_obj = gdf_POI_2_tile_ids[(gdf_POI_2_tile_ids['tag'] == type_POI) &
                    (gdf_POI_2_tile_ids['name'] == spatial_unit ) & 
                    (gdf_POI_2_tile_ids['type'] == f"{POI_or_station}{expanded}")
    ]
    assert len(gdf_obj) == 1, f"Length of gdf = {len(gdf_obj)} while it should be = 1"

    osmid = gdf_obj['id'].values[0]
    path_df = f"{FOLDER_PATH}/POIs/netmob_POI_Lyon{expanded}/{type_POI}/{app}/df_{osmid}_{transfer_mode}.csv"
    serie = pd.read_csv(path_df,index_col = 0).sum(axis=1)
    serie.index = pd.to_datetime(serie.index)
    return(serie)


def analysis_on_specific_training_mode(trainer,ds,training_mode,transfer_modes= None,
                                       type_POIs = ['stadium','nightclub'],
                                       spatial_units = ['Lou_rugby','Ninkasi_Kao'],
                                       apps = ['Instagram'],
                                       POI_or_stations = ['POI'],
                                       expanded = '',
                                       station = 'BON'
                                       ):

    Preds,Y_true,T_labels = trainer.testing(ds.normalizer, training_mode =training_mode)                                  
    df_true,df_prediction = get_df_for_visualisation(ds,Preds,Y_true,training_mode)
    kick_off_time,match_times = rugby_matches(df_true.index,RANGE)

    if apps is not None : 
        # Load gdf for POIs:
        gdf_POI_2_tile_ids = gpd.read_file(f"{FOLDER_PATH}/POIs/gdf_POI_2_tile_ids.geojson")
        netmob_consumption = pd.DataFrame(index = df_true.index)
        for app in apps:
            for type_POI,spatial_unit,POI_or_station in zip(type_POIs,spatial_units,POI_or_stations):
                for transfer_mode in transfer_modes:
                    serie_netmob = netmob_volume_on_POI(gdf_POI_2_tile_ids,app,transfer_mode,type_POI,spatial_unit,POI_or_station,expanded)
                    serie_netmob = serie_netmob.loc[df_true.index]

                    # norm_series :
                    serie_netmob = (serie_netmob-serie_netmob.min())/(serie_netmob.max()-serie_netmob.min())
                    
                    name_netmob_serie = f"{app}_{transfer_mode} at {spatial_unit}"

                    netmob_consumption[name_netmob_serie] = serie_netmob
        netmob_consumption['Sum_of_apps'] = netmob_consumption.sum(axis=1)/len(netmob_consumption.columns)
    else:
        netmob_consumption = None

    visualisation_special_event(trainer,df_true,df_prediction,station,kick_off_time,RANGE,WIDTH,HEIGHT,MIN_FLOW,training_mode = training_mode,netmob_consumption = netmob_consumption)

# Get df_True Volume: 
def get_df_for_visualisation(ds,Preds,Y_true,training_mode):
       '''
       outputs:
       --------
       return 2 pd DataFrame : df_true and df_prediction
       >>>> the DataFrames contains the unormalized predicted and real value  
       '''
       df_verif = getattr(ds.tensor_limits_keeper,f"df_verif_{training_mode}")  
       df_true = pd.DataFrame(Y_true[:,:,0],columns = ds.spatial_unit,index = df_verif.iloc[:,-1])
       df_prediction = pd.DataFrame(Preds[:,:,0],columns = ds.spatial_unit,index = df_verif.iloc[:,-1])
       return(df_true,df_prediction)


def visualisation_special_event(trainer,df_true,df_prediction,station,kick_off_time,Range,width,height,min_flow,training_mode,netmob_consumption):
    ''' Specific interactiv visualisation for Prediction, True Value, Error and loss function '''
    p1 = plot_single_point_prediction(df_true,df_prediction,station,title= f'{training_mode} Trafic Volume Prediction at each subway station ',kick_off_time=kick_off_time, range=Range,width=width,height = height,bool_show = False)
    p2 = plot_TS(netmob_consumption,width=width,height=height,bool_show=False) if netmob_consumption is not None else None
    p3 = plot_prediction_error(df_true,df_prediction,station,metrics =['mae','mse','mape'],title = 'Prediction Error',width=width,height=height,bool_show=False,min_flow = min_flow)

    select = drag_selection_box(df_true,p1,p2,p3,width=width,height=height//3)
    output_notebook()
    if p2 is not None:
        col1 = column(p1,p2,p3,select)
    else: 
        col1 = column(p1,p3,select)     

    col2 = plot_loss_from_trainer(trainer,width=width//3,height=height,bool_show=False)
    grid = row(col1,col2)

    show(grid)

def get_ds(model_name,dataset_names,dataset_for_coverage,
           modification = {},
            args_init = None
            ):
    save_folder = None
    df_loss,df_results = pd.DataFrame(),pd.DataFrame()


    # Tricky but here we net to set 'netmob' so that we will use the same period for every combination
    args,folds,hp_tuning_on_first_fold = local_get_args(model_name,
                                                        args_init,
                                                            dataset_names=dataset_names,
                                                            dataset_for_coverage=dataset_for_coverage,
                                                            modification = modification)
    K_fold_splitter,K_subway_ds,args_with_contextual = get_inputs(args,folds)
    args_with_contextual = modification_contextual_args(args_with_contextual,modification)
    trial_id = get_trial_id(args_with_contextual)
    ds = K_subway_ds[0]
    return(ds,args_with_contextual,trial_id,save_folder,df_loss)