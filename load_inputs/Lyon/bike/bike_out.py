import sys
import os
import pandas as pd
import torch
import numpy as np
from datetime import datetime

# --- Gestion de l'arborescence ---
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- Importations personnalisées ---
from pipeline.dataset import DataSet, PersonnalInput
from pipeline.utils.utilities import filter_args # Assurez-vous que ce chemin est correct
from pipeline.build_inputs.load_preprocessed_dataset import load_input_and_preprocess
from load_inputs.Lyon.bike.bike_in import load_data as load_data_in
# --- Constantes spécifiques à cette donnée ---
NAME = 'bike_out'
FILE_BASE_NAME = 'velov'
DIRECTION = 'emitted' # attracted
FILE_PATTERN = f'{FILE_BASE_NAME}_{DIRECTION}_by_station' # Sera complété par args.freq
DATA_SUBFOLDER = f'agg_data/{FILE_BASE_NAME}' # Sous-dossier dans FOLDER_PATH


# Couverture théorique
START = '2019-01-01'
END = '2020-01-01'
USELESS_DATES = {'hour':[], #[1,2,3,4,5,6],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }
# Liste des périodes invalides
list_of_invalid_period = []

C = 1 # Nombre de canaux/features par unité spatiale

# Colonnes attendues dans le CSV
DATE_COL = 'date_sortie' # Ou 'date_entree' pour 'attracted'?
LOCATION_COL = 'id_sortie' # Ou 'id_entree' pour 'attracted'?
VALUE_COL = 'volume'



def load_data(FOLDER_PATH, coverage_period, invalid_dates, args, minmaxnorm,standardize, normalize=True,
              tensor_limits_keeper = None):
    return load_data_in(FOLDER_PATH, coverage_period, invalid_dates, args, minmaxnorm,standardize, normalize=normalize,
                        tensor_limits_keeper = tensor_limits_keeper,
            file_pattern = FILE_PATTERN,
              data_subfolder = DATA_SUBFOLDER,
              date_col = DATE_COL,
              location_col = LOCATION_COL,
              value_col = VALUE_COL,
              direction = DIRECTION,
              name = NAME)