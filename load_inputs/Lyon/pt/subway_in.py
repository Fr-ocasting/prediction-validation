import sys 
import os 
import pandas as pd
import numpy as np
import torch
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from pipeline.dataset import DataSet
from datetime import datetime 
from pipeline.utils.utilities import filter_args,get_time_step_per_hour,restrain_df_to_specific_period,remove_outliers_based_on_quantile
from pipeline.build_inputs.load_preprocessed_dataset import load_input_and_preprocess

''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - add argument 'num_nodes', 'C' to the NameSpace. These are specific to this data
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''
NAME= 'subway_in'
FILE_NAME = 'subway_in/subway_in'#'subway_IN_interpol_neg_15_min_2019_2020' #.csv
START = '01/01/2019'
END = '01/01/2020'
FREQ = '15min'
USELESS_DATES = {'hour':[1,2,3,4,5,6],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }

list_of_invalid_period = []
list_of_invalid_period.append([datetime(2019,1,10,15,30),datetime(2019,1,14,15,30)])
list_of_invalid_period.append([datetime(2019,1,30,8,15),datetime(2019,1,30,10,30)])
list_of_invalid_period.append([datetime(2019,2,18,11),datetime(2019,2,18,13)])
list_of_invalid_period.append([datetime(2019,4,23,14),datetime(2019,4,28,14)])
list_of_invalid_period.append([datetime(2019,6,26,11),datetime(2019,6,28,4)])
list_of_invalid_period.append([datetime(2019,10,27),datetime(2019,10,28,20,15)])
list_of_invalid_period.append([datetime(2019,12,21,15,45),datetime(2019,12,21,16,45)])


C = 1
num_nodes = 40

def load_data(FOLDER_PATH,invalid_dates,coverage_period,args,minmaxnorm,standardize,normalize= True,filename=None,name=NAME,tensor_limits_keeper = None):
    dataset = load_DataSet(args,FOLDER_PATH,coverage_period = coverage_period,filename=filename,name=name)
    args_DataSet = filter_args(DataSet, args)

    if  hasattr(args,'contextual_kwargs') and (name in args.contextual_kwargs.keys()) and ('use_future_values' in args.contextual_kwargs[name].keys()) and args.contextual_kwargs[name]['use_future_values'] and ('loading_contextual_data' in args.contextual_kwargs[name].keys()) and args.contextual_kwargs[name]['loading_contextual_data']:
        data_T = torch.roll(torch.Tensor(dataset.raw_values), shifts=-1, dims=0)
        print(f">>>>> ICI ON UTILISE LE {name.upper()} IN FUTURE !!!!")
        print('data_T.size: ',data_T.size())
    else:
        data_T = dataset.raw_values

    preprocesed_ds = load_input_and_preprocess(dims = dataset.dims,normalize=normalize,invalid_dates=invalid_dates,args=args,data_T=data_T,
                                            coverage_period=coverage_period,name=name,
                                            minmaxnorm=minmaxnorm,standardize=standardize,
                                            tensor_limits_keeper=tensor_limits_keeper)
    
    preprocesed_ds.spatial_unit = dataset.spatial_unit
    preprocesed_ds.dims = dataset.dims
    preprocesed_ds.periods = dataset.periods
    preprocesed_ds.time_step_per_hour = dataset.time_step_per_hour
    preprocesed_ds.indices_spatial_unit = dataset.indices_spatial_unit
    preprocesed_ds.city = dataset.city


    return preprocesed_ds


def load_DataSet(args,FOLDER_PATH,coverage_period = None,filename=None,name = NAME):
    '''Load the dataset. Supposed to coontains pd.DateTime Index as index, and named columns.
    columns has to represent the spatial units.

    outputs: 
    ---------
    df: contains 
    df.index : coverage period of the dataset 
    invalid_dates : list of invalid dates 
    '''
    if filename==None:
        filename = FILE_NAME

    df = load_subway_in_df(args,FOLDER_PATH,filename=filename,
                           coverage_period = coverage_period,
                           name=name)

    if (hasattr(args,'set_spatial_units')) and (args.set_spatial_units is not None) :
        print('   Number of Considered Spatial-Unit: ',len(args.set_spatial_units))
        spatial_unit = args.set_spatial_units
        indices_spatial_unit = [list(df.columns).index(station_i) for station_i in  spatial_unit]
        df = df[spatial_unit]
    else:
        spatial_unit = df.columns
        indices_spatial_unit = np.arange(len(df.columns))

    time_step_per_hour = get_time_step_per_hour(args.freq)
    weekly_period =  int((24-len(USELESS_DATES['hour']))*(7-len(USELESS_DATES['weekday']))*time_step_per_hour)
    daily_period =  int((24-len(USELESS_DATES['hour']))*time_step_per_hour)
    periods = [weekly_period,daily_period]  

    args_DataSet = filter_args(DataSet, args)
    if 'time_step_per_hour' in args_DataSet.keys():
        del args_DataSet['time_step_per_hour']
    


    dataset = DataSet(df,
                      time_step_per_hour=time_step_per_hour, 
                      spatial_unit = spatial_unit,
                      indices_spatial_unit = indices_spatial_unit,
                      dims = [0],
                      city = 'Lyon',
                      periods = periods,
                      **args_DataSet)

    return(dataset)
    

def load_subway_in_df(args,FOLDER_PATH,filename,coverage_period,name=NAME):
    
    print(f"   Load data from: /{FOLDER_PATH}/{filename}.csv")
    try:
        df = pd.read_csv(f"{FOLDER_PATH}/{filename}.csv",index_col = 0)
    except:
        raise FileNotFoundError(f"   ERROR : File {FOLDER_PATH}/{filename}.csv has not been found.")
    
    df.columns.name = 'Station'
    df.index = pd.to_datetime(df.index)

    # Remove ouliers
    df = remove_outliers(df,args,name)


    if args.freq != FREQ :
        if args.freq[-1] == 'H': 
            freq_i = int(args.freq.replace('H',''))*60
        else:
            freq_i = int(args.freq.replace('min',''))
        assert int(freq_i)> int(FREQ.replace('min','')), f'Trying to apply a a {args.freq} temporal aggregation while the minimal possible one is {FREQ}'
        df = df.resample(args.freq).sum()

    # Temporal Restriction: 
    df = restrain_df_to_specific_period(df,coverage_period)
    # Restrain to specific stations:
    df_correspondance = get_trigram_correspondance()
    df_correspondance.sort_values(by = 'Station',inplace = True)
    df = df.rename(columns = {row.Station: row.COD_TRG for _,row in df_correspondance.iterrows()})
    df = df[df_correspondance.COD_TRG]
    # ...
    return df
  
def remove_outliers(df,args,name):
    '''
    Replace the outliers by linear interpolation. Outliers are identified as MaxiMum flow recorded during the 'light festival' in Lyon. 
    It's an atypical event which reach the highest possible flow. Having higher flow on passenger is almost impossible.
    '''
    limits = {
        'BEL': 2700,
        'CHA': 1700,
        'GOR': 1700
    }
    default_limit = 1500

    # Appliquer les limites
    for column in df.columns:
        limit = limits.get(column, default_limit)
        df[column] = df[column].where(df[column] <= limit, None)

    # Interpolation linéaire
    df_interpolated = df.interpolate(method='linear')

    # Remplacer les valeurs originales par les interpolées
    df.update(df_interpolated)

    # # remove outliers by quantile filtering. But differenciate according if it's for contextual dataset or target dataset:
    df = remove_outliers_based_on_quantile(df,args,name)

    return df


def get_trigram_correspondance():
    ''' Some surprise : 
        Vieux Lyon : Jea
        Gare d'oulins : OGA
    '''
    df = pd.DataFrame(columns = ['Station','COD_TRG'])
    df['COD_TRG'] = ['AMP','BEL','BRO','COR',
                     'CUI','CUS','FLA','GOR',
                     'BLA','GRA','GUI','GIL',
                     'HEN','HOT','LAE','MAS',

                     'MER','LUM','PRY','PER',
                     'SAN','SAX','VMY','JEA', #JEA = Vieux Lyon
                     'BON','CHA','VAI','VEN',

                     'MAC','GAR','FOC','REP',
                     'GER','DEB','JAU','CPA',
                     'CRO','PAR','SOI','OGA']
    
    df['Station'] =['Ampère Victor Hugo','Bellecour','Brotteaux','Cordeliers',
                    'Cuire','Cusset','Flachet','Gorge de Loup',
                    'Grange Blanche','Gratte Ciel','Place Guichard','Guillotière',
                    'Hénon','Hôtel de ville - Louis Pradel','Laënnec','Masséna',

                    'Mermoz - Pinel','Monplaisir Lumière','Parilly','Perrache',
                    'Sans Souci','Saxe - Gambetta','Valmy','Vieux Lyon',
                    'Laurent Bonnevay','Charpennes','Gare de Vaise','Gare de Vénissieux',

                    'Jean Macé','Garibaldi','Foch','République Villeurbanne',
                    'Stade de Gerland','Debourg','Place Jean Jaurès','Croix Paquet',
                    'Croix-Rousse','Part-Dieu','La soie',"Gare d'Oullins"]
    
    df['INDIV'] = ['AMPERE', 'BELLECOUR', 'BROTTEAUX','CORDELIERS',
                   'CUIRE', 'CUSSET','FLACHET','GORGE DE LOUP',
                   'GRANGE BLANCHE', 'GRATTE CIEL', 'PLACE GUICHARD', 'GUILLOTIERE',
                    'HENON', 'HOTEL DE VILLE', 'LAENNEC','MASSENA',      

                    'MERMOZ PINEL', 'MONPLAISIR LUMIERE','PARILLY','PERRACHE',
                    'SANS SOUCI', 'SAXE GAMBETTA','VALMY','VIEUX LYON',
                    'LAURENT BONNEVAY', 'CHARPENNES', 'GARE DE VAISE', 'GARE DE VENISSIEUX',

                    'JEAN MACE','GARIBALDI','FOCH', 'REPUBLIQUE',
                    'STADE DE GERLAND', 'DEBOURG', 'PLACE JEAN JAURES','CROIX PAQUET',
                    'CROIX ROUSSE', 'PART DIEU',  'VAULX-EN-VELIN LA SOIE', 'OULLINS GARE',]
    return(df)