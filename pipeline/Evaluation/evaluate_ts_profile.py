# GET PARAMETERS
import sys
import os
import pandas as pd
import numpy as np 
import geopandas as gpd 
import torch
if torch.cuda.is_available():
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32  = True
   
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    

from pipeline.plotting.TS_analysis import drag_selection_box,plot_single_point_prediction,plot_prediction_error,plot_loss_from_trainer,plot_TS
from bokeh.plotting import show,output_notebook
from bokeh.layouts import column,row
from pipeline.utils.specific_event import rugby_matches
from constants.paths import FOLDER_PATH
from pipeline.utils.loger import LOG
from experiences.loop_train_save_log import train_one_config
from pipeline.DataSet.load_datasets import get_ds_without_shuffling_on_train_set

RANGE = 3*60  # +/- range (min) supposed to be affected around the event 
WIDTH = 1000
HEIGHT = 300




def evaluate_config(config_init = None,
                    modification = {},
                    fold_to_evaluate = None,
                    training_mode_to_visualise = ['test','valid','train'],
                    station=['GER'],
                    transfer_modes= None,
                    type_POIs = None,
                    spatial_units = None,
                    apps = None,
                    POI_or_stations = None,
                    expanded =None,
                    individual_poi = True,
                    sum_ts_pois = True,
                    save_folder= None,
                    trial_id = None, 
                    ):
    
    '''
    args: 
    type_POIs : list of type of POIs.                         >>> ['stadium','nightclub']
    spatial_units : list of name of spatial units to analyse. >>> ['Matmut Stadium Gerland']
    apps : list of apps to deal with.                         >>> ['Instagram']
    POI_or_stations : list of type of object to analyse contains occurence of 'POI' or 'station'.  >>> ['POI']
    transfer_modes : list of considered transfer modes (downlink / uplink.   >>> ['DL','UL']
    expanded: '' if we look at the intensity of netmob consumption at the POI. '_expanded' if we look also one square around.
    '''
    loger = LOG()
    trainer,ds,model,args_init,loger = train_one_config(
                                                        config_i=config_init,
                                                        init_save_folder=save_folder,
                                                        trial_id=trial_id,
                                                        loger = loger,
                                                        modification = modification,

                                                        )

    print(f"\n--------------------------------------------------")
    print(f"Reload dataset without shuffling on train set, and remove data_augmentation")
    trainer,ds_no_shuffle = get_ds_without_shuffling_on_train_set(trainer,modification,args_init,fold_to_evaluate)


    for training_mode in training_mode_to_visualise:
        analysis_on_specific_training_mode(trainer,ds_no_shuffle,
                                           training_mode=training_mode,
                                           transfer_modes= transfer_modes,
                                           type_POIs = type_POIs,
                                           spatial_units=spatial_units,
                                           apps=apps,
                                           POI_or_stations = POI_or_stations,
                                           expanded=expanded,
                                           station=station,
                                           individual_poi = individual_poi, 
                                           sum_ts_pois =sum_ts_pois )
    return(trainer,ds,ds_no_shuffle,args_init)





def netmob_volume_on_POI(gdf_POI_2_tile_ids,app = 'Instagram',transfer_mode = 'DL',type_POI = 'stadium', spatial_unit = 'Lou_rugby',POI_or_station='POI',expanded=''):
    gdf_obj = gdf_POI_2_tile_ids[(gdf_POI_2_tile_ids['tag'] == type_POI) &
                    (gdf_POI_2_tile_ids['name'] == spatial_unit ) & 
                    (gdf_POI_2_tile_ids['type'] == f"{POI_or_station}{expanded}")
    ]
    #print('type_POI: ',type_POI,'spatial_unit: ', spatial_unit, 'POI_or_station: ',POI_or_station)
    assert len(gdf_obj) == 1, f"Length of gdf = {len(gdf_obj)} while it should be = 1"

    osmid = gdf_obj['id'].values[0]
    path_df = f"{FOLDER_PATH}/POIs/netmob_POI_Lyon{expanded}/{type_POI}/{app}/df_{osmid}_{transfer_mode}.csv"
    serie = pd.read_csv(path_df,index_col = 0).sum(axis=1)
    serie.index = pd.to_datetime(serie.index)
    return(serie)


def analysis_on_specific_training_mode(trainer,ds,training_mode,transfer_modes= None,
                                       type_POIs = ['stadium','nightclub'],
                                       spatial_units = ['Lou_rugby','Ninkasi_Kao'],
                                       apps = None, #['Instagram'],
                                       POI_or_stations = ['POI'],
                                       expanded = '',
                                       station = 'BON',
                                       individual_poi = True,
                                       sum_ts_pois = True,
                                       min_flow = None,
                                       metrics = ['mae'],
                                       width = WIDTH,
                                       height = HEIGHT,
                                       title = None,
                                       ):
    '''
    individual_poi: if True then each Time-Serie of each POIs is visualised
    sum_ts_pois: if True then for each app and each transfer mode, the sum of all the POIs are represented
    '''

    if hasattr(trainer,'best_weights'):
        trainer.model.load_state_dict(trainer.best_weights, strict=True)

    Preds,Y_true,T_labels = trainer.testing(ds.normalizer, training_mode =training_mode)                                  
    df_true,df_predictions = get_df_for_visualisation(ds,Preds,Y_true,training_mode,trainer.out_dim_factor,stations=station)
    kick_off_time,match_times = rugby_matches(df_true.index,RANGE)

    if apps is not None : 
        netmob_consumption = get_netmob_consumption_on_specifics_tags_apps(df_true.index,apps,type_POIs,spatial_units,POI_or_stations,transfer_modes,expanded, individual_poi, sum_ts_pois)
    else:
        netmob_consumption = None
    visualisation_special_event(trainer,df_true,df_predictions,station,kick_off_time,RANGE,width,height,training_mode = training_mode,netmob_consumption = netmob_consumption,min_flow = min_flow,metrics = metrics, title=title)

def get_netmob_consumption_on_specifics_tags_apps(s_dates,apps,type_POIs,spatial_units,POI_or_stations,transfer_modes,expanded, individual_poi = True, sum_ts_pois = True):
    # Load gdf for POIs:
    gdf_POI_2_tile_ids = gpd.read_file(f"{FOLDER_PATH}/POIs/gdf_POI_2_tile_ids.geojson")
    netmob_consumption = pd.DataFrame(index = s_dates)
    for app in apps:
        for transfer_mode in transfer_modes:
            netmob_consumption_app_i = pd.DataFrame(index = s_dates)
            for type_POI,spatial_unit,POI_or_station in zip(type_POIs,spatial_units,POI_or_stations):
            
                serie_netmob = netmob_volume_on_POI(gdf_POI_2_tile_ids,app,transfer_mode,type_POI,spatial_unit,POI_or_station,expanded)
                serie_netmob = serie_netmob.loc[s_dates]

                # norm_series :
                serie_netmob = (serie_netmob-serie_netmob.min())/(serie_netmob.max()-serie_netmob.min())
                
                name_netmob_serie = f"{app}_{transfer_mode} at {spatial_unit}"
                if individual_poi:
                    netmob_consumption[name_netmob_serie] = serie_netmob
                netmob_consumption_app_i[name_netmob_serie] = serie_netmob
            if sum_ts_pois:
                netmob_consumption[f'{app}_{transfer_mode}_sum_POIs'] = netmob_consumption_app_i.sum(axis=1)
    #netmob_consumption['Sum_of_apps'] = netmob_consumption.sum(axis=1)/len(netmob_consumption.columns)
    return netmob_consumption

# Get df_True Volume: 
def get_df_for_visualisation(ds,Preds,Y_true,training_mode,out_dim_factor,stations):
    '''
    outputs:
    --------
    return 2 pd DataFrame : df_true and df_prediction
    >>>> the DataFrames contains the unormalized predicted and real value  
    '''

    df_verif = getattr(ds.tensor_limits_keeper,f"df_verif_{training_mode}")  
    #df_true = pd.DataFrame(Y_true[:,:,-1],columns = ds.spatial_unit,index = df_verif.iloc[:,-1])
    
    nb_step_ahead = Preds.size(-1)//out_dim_factor
    if stations is not None:
        remaining_column_ind = [k for k,c in enumerate(ds.spatial_unit) if c in stations]
    else:
        stations = ds.spatial_unit
        remaining_column_ind = np.arange(len(ds.spatial_unit))

    sub_Preds = Preds[:,remaining_column_ind,:].detach().cpu().numpy()
    sub_Y_true = Y_true[:,remaining_column_ind,-1].detach().cpu().numpy()

    def get_multi_index_columns(stations,L_out_dim_factor,L_nb_step_ahead):
        level_0,level_1,level_2 = [],[],[]
        for station in stations:
            for q in L_out_dim_factor:
                for h in L_nb_step_ahead:
                    level_0.append(station)
                    level_1.append(f'h{h+1}')
                    level_2.append(f'q{q}')
        multi_index = pd.MultiIndex.from_tuples(list(zip(level_0,level_1,level_2)),names=['station','step_ahead','q'])
        return multi_index


    df_true =  pd.DataFrame(sub_Y_true,columns = stations,index = df_verif.iloc[:,-1])

    df_predictions = []
    for i in range(out_dim_factor):
        for sa in range(nb_step_ahead):
            multi_index = get_multi_index_columns(stations,[i],[sa])
            df_predictions.append(pd.DataFrame(sub_Preds[:,:,nb_step_ahead*i + sa],
                                            columns = multi_index,
                                            index = df_verif.iloc[:,-nb_step_ahead+sa])
            )
    #df_predictions = [pd.DataFrame(Preds[:,:,output_i],columns = ds.spatial_unit,index = df_verif.iloc[:,-1]) for output_i in range(Preds.size(-1))]
    return(df_true,df_predictions)

def visualisation_special_event(trainer,
                                df_true,
                                df_prediction,
                                station,
                                kick_off_time=[],
                                Range=None,
                                width=1200,
                                height=300,
                                min_flow=None,
                                training_mode='test',
                                netmob_consumption=None,
                                metrics = ['mae'],
                                title = None):
    ''' Specific interactiv visualisation for Prediction, True Value, Error and loss function '''
    p1 = plot_single_point_prediction(df_true,df_prediction,station,
                                      title= f'{training_mode} Trafic variable Prediction at each spatial units ' if title is None else title,
                                      kick_off_time=kick_off_time, 
                                      range=Range,
                                      width=width,
                                      height = height,
                                      bool_show = False,
                                      out_dim_factor=trainer.out_dim_factor,
                                      nb_step_ahead=trainer.step_ahead,
                                      horizon_step = trainer.horizon_step,
                                      freq=trainer.args.freq,)
    
    p2 = plot_TS(netmob_consumption,width=width,height=height,bool_show=False) if netmob_consumption is not None else None

    if (df_prediction is not None) and (len(df_prediction)==1):
        p3 = plot_prediction_error(df_true,df_prediction[0],station,metrics =metrics,title = 'Prediction Error',width=width,height=height,bool_show=False,min_flow = min_flow)
    else:
        p3=None

    select = drag_selection_box(df_true,p1,p2,p3,width=width,height=height//3)
    if p2 is not None:
        if p3 is not None:
            col1 = column(p1,p2,p3,select)
        else:
            col1 = column(p1,p2,select)
    else: 
        if p3 is not None:
            col1 = column(p1,p3,select)   
        else:
            col1 = column(p1,select) 
    height_loss = int((2+1/3)*height) if p2 is not None else height
    col2 = plot_loss_from_trainer(trainer,width=width//2,height=height_loss,bool_show=False)
    if col2 is None:
        grid = col1
    else:
        grid = row(col1,col2)

    output_notebook()
    show(grid)

