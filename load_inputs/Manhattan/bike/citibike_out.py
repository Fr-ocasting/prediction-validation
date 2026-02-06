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
from load_inputs.Manhattan.bike.citibike_in import load_data as load_data_in
from load_inputs.Manhattan.bike.citibike_in import START, END

# --- Constantes spécifiques à cette donnée ---
NAME = 'citibike_out'
FILE_BASE_NAME = 'citibike'
DIRECTION = 'emitted' # attracted
CITY = 'Manhattan'

# Couverture théorique
# START =  load_inputs.Manhattan.bike.bike_in.START
# END   =  load_inputs.Manhattan.bike.bike_in.END
USELESS_DATES = {'hour':[], #[1,2,3,4,5,6],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }
# Liste des périodes invalides
list_of_invalid_period = []

C = 1 # Nombre de canaux/features par unité spatiale


def load_data(FOLDER_PATH, coverage_period, invalid_dates, args, minmaxnorm,standardize, normalize=True,
              tensor_limits_keeper = None):
    
    return load_data_in(FOLDER_PATH, coverage_period, invalid_dates, args, minmaxnorm,standardize, normalize=normalize,
              tensor_limits_keeper = tensor_limits_keeper,
              direction = DIRECTION,
              name = NAME)