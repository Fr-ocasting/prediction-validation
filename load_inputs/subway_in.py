import sys 
import os 
import pandas as pd
import numpy as np
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from dataset import DataSet
from datetime import datetime 

''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - add argument 'n_vertex', 'C' to the NameSpace. These are specific to this data
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''

FILE_NAME = 'subway_in/subway_in'#'subway_IN_interpol_neg_15_min_2019_2020' #.csv

list_of_invalid_period = []
list_of_invalid_period.append([datetime(2019,1,10,15,30),datetime(2019,1,14,15,30)])
list_of_invalid_period.append([datetime(2019,1,30,8,15),datetime(2019,1,30,10,30)])
list_of_invalid_period.append([datetime(2019,2,18,11),datetime(2019,2,18,13)])
list_of_invalid_period.append([datetime(2019,4,23,14),datetime(2019,4,28,14)])
list_of_invalid_period.append([datetime(2019,6,26,11),datetime(2019,6,28,4)])
list_of_invalid_period.append([datetime(2019,10,27),datetime(2019,10,28,16)])
list_of_invalid_period.append([datetime(2019,12,21,15,45),datetime(2019,12,21,16,45)])

INVALID_DATES = []
for start,end in list_of_invalid_period:
    INVALID_DATES = INVALID_DATES + list(pd.date_range(start,end,freq = f'15min'))
C = 1
n_vertex = 40
COVERAGE = pd.date_range(start='01/01/2019', end='01/01/2020', freq='15min')[:-1]


def load_data(args,ROOT,FOLDER_PATH,coverage_period = None):
    '''Load the dataset. Supposed to coontains pd.DateTime Index as index, and named columns.
    columns has to represent the spatial units.

    outputs: 
    ---------
    df: contains 
    df.index : coverage period of the dataset 
    invalid_dates : list of invalid dates 
    '''

    df = pd.read_csv(f"{ROOT}/{FOLDER_PATH}/{FILE_NAME}.csv",index_col = 0)
    df.columns.name = 'Station'
    df.index = pd.to_datetime(df.index)

    df = restrain_df_to_specific_period(df,coverage_period)
    time_step_per_hour = (60*60)/(df.iloc[1].name - df.iloc[0].name).seconds
    assert time_step_per_hour == 4, 'TIME STEP PER HOUR = {time_step_per_hour} ALORS QU ON VEUT =4 '

    df_correspondance = get_trigram_correspondance()
    df_correspondance.set_index('Station').reindex(df.columns)
    df.columns = df_correspondance.COD_TRG

    # Remove ouliers
    df = remove_outliers(df)

    if (hasattr(args,'set_spatial_units')) and (args.set_spatial_units is not None) :
        print('Considered Spatial-Unit: ',args.set_spatial_units)
        spatial_unit = args.set_spatial_units
        indices_spatial_unit = [list(df.columns).index(station_i) for station_i in  spatial_unit]
        df = df[spatial_unit]
    else:
        spatial_unit = df.columns
        indices_spatial_unit = np.arange(len(df.columns))


    dataset = DataSet(df,
                      time_step_per_hour=time_step_per_hour, 
                      Weeks = args.W, 
                      Days = args.D, 
                      historical_len= args.H,
                      step_ahead=args.step_ahead,
                      spatial_unit = spatial_unit,
                      indices_spatial_unit = indices_spatial_unit,
                      dims = [0],
                      city = 'Lyon'
                      )
    
    return(dataset)
    

def remove_outliers(df):
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
    return df

def restrain_df_to_specific_period(df,coverage_period):
    if coverage_period is not None:
        df = df.loc[coverage_period]

    df = df.sort_index()
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
                     'SAN','SAX','VMY','JEA',
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
    return(df)