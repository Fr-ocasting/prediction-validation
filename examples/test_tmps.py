

import os 
import sys
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from examples.train_and_visu_non_recurrent import evaluate_config
dataset_names = ["calendar"] # ["subway_in","calendar"] # ["subway_in"] # ['data_bidon'] # ['METR_LA'] # ['PEMS_BAY']
dataset_for_coverage = ['subway_in','netmob_image_per_station'] #  ['data_bidon','netmob'] #  ['subway_in','netmob']  # ['METR_LA'] # ['PEMS_BAY']
model_name = None

station = 'GER'  # 'BON'  #'GER'
apps = ['Google_Maps']# ['Instagram','Twitter','Google_Maps'] # 'Instagram'  # 'Twitter' # 'Google_Maps' # 'Facebook'


transfer_modes = ['DL'] # ['DL'] # ['UL'] # ['DL','UL']
type_POIs = ['stadium'] #['stadium','station'] #['stadium','nightclub'] #['stadium']
spatial_units = ['Matmut Stadium Gerland','GER']  #spatial_units = ['Lou_rugby']  # ['Astroballe'] #['Lou_rugby','Ninkasi_Kao'] #['Lou_rugby'] #['Ninkasi_Kao'] 
expanded = '_expanded' # '' # '_expanded' # ''
POI_or_stations = ['POI']# ['POI','station'] # 'station'
modification = {'epochs' : 5, #100
                'lr':5e-4,
                'set_spatial_units': ['GER'],#['CHA','GER','BON','SOI'],
                'TE_concatenation_late':False,
                'TE_concatenation_early':True,   
                'TE_embedding_dim' : 1             
                           }
(trainer,ds,ds_no_shuffle,args) = evaluate_config(model_name,dataset_names,dataset_for_coverage,transfer_modes= transfer_modes,
                                                   type_POIs = type_POIs,spatial_units = spatial_units,apps = apps,POI_or_stations = POI_or_stations,
                                                   expanded=expanded,modification=modification)
