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
from dataset import DataSet, PersonnalInput
from utils.utilities import filter_args # Assurez-vous que ce chemin est correct
from build_inputs.load_preprocessed_dataset import load_input_and_preprocess
# --- Constantes spécifiques à cette donnée ---
# Le nom de fichier sera construit dynamiquement basé sur args.freq
NAME = 'subway_indiv'
FILE_BASE_NAME = 'subway_indiv'
DATA_SUBFOLDER = 'agg_data/validation_individuelle' # Sous-dossier dans FOLDER_PATH

# Fréquence native la plus fine disponible (pour info, le chargement dépendra de args.freq)
NATIVE_FREQ = '3min'
# Couverture théorique (à remplacer par les vraies dates si connues)
START = '2019-11-01' # Exemple basé sur head()
END = '2020-03-30 23:30:00'

# Liste des périodes invalides (à compléter si nécessaire)
list_of_invalid_period = []
USELESS_DATES = {'hour':[1,2,3,4,5,6],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }
C = 1 # Nombre de canaux/features par unité spatiale

# Colonnes attendues dans le CSV
DATE_COL = 'VAL_DATE'
LOCATION_COL = 'COD_TRG'
VALUE_COL = 'Flow'
def load_data(FOLDER_PATH, invalid_dates, coverage_period, args,minmaxnorm,standardize, normalize=True,
                data_subfolder = DATA_SUBFOLDER,
                file_base_name = FILE_BASE_NAME
              ):
    """
    Charge, pivote, filtre et pré-traite les données subway_indiv.

    Args:
        dataset (DataSet): L'objet DataSet principal (pour contexte).
        FOLDER_PATH (str): Chemin vers le dossier contenant data_subfolder.
        invalid_dates (list): Liste des périodes invalides fournie globalement.
        coverage_period (pd.DatetimeIndex): Période temporelle à conserver.
        args (Namespace): Arguments globaux (contient args.freq).
        normalize (bool): Faut-il normaliser les données.

    Returns:
        PersonnalInput: Objet contenant les données traitées.
    """
    target_freq = args.freq
    file_name = f"{file_base_name}_{target_freq}"
    data_file_path = os.path.join(FOLDER_PATH, data_subfolder, file_name, f"{file_name}.csv")

    print(f"   Load data from: {data_file_path}")
    try:
        df = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print(f"   ERROR : File {data_file_path} has not been found.")
        print(f"   Check if '{target_freq}' exists in {file_base_name} and than paths are well set.")
        return None
    except Exception as e:
        print(f"   ERROR while loading {file_name}.csv: {e}")
        return None

    # --- Prétraitement ---
    try:
        
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df = df.set_index(DATE_COL)
        reindex = pd.date_range(start=START, end=END, freq=args.freq)[:-1]
        df = df.reindex(reindex).fillna(0)


        df_reindexed = df[df.index.isin(coverage_period)].copy()

        if df_reindexed.empty:
             print(f"   ERROR : Not any remainig data in {file_name}.csv")
             print(f"   Check the current coverage period on the trial: ({min(coverage_period)} - {max(coverage_period)})")
             if df is not None:
                print(f"   And the maximum coverage period of {file_name}: ({df.index.min()} - {df.index.max()})")
             else:
                print(f"   DataFrame df is empty, no data in {file_name}.csv")
        data_T = torch.tensor(df_reindexed.values).float()

    except KeyError as e:
        print(f"   ERROR: Missing column within {file_name}.csv : {e}. Check if columns DATE_COL, LOCATION_COL, VALUE_COL exists.")
        return None
    except Exception as e:
        print(f"   ERROR while pre-processing {file_name}.csv: {e}")
        return None


    # --- Création et Prétraitement avec PersonnalInput ---
    # Utilisation de la fonction helper locale
    #print("   Création et prétraitement de l'objet PersonnalInput...")
    dims = [0] # if [0] then Normalisation on temporal dim

    processed_input = load_input_and_preprocess(dims = dims,normalize=normalize,invalid_dates=invalid_dates,args=args,data_T=data_T,coverage_period=coverage_period,name=NAME,
                                                minmaxnorm=minmaxnorm,standardize=standardize)

    # --- Finalisation Métadonnées ---
    processed_input.spatial_unit = df_reindexed.columns.tolist()
    processed_input.C = C
    processed_input.periods = None # Pas de périodicité spécifique définie ici
    return processed_input


if __name__ == "__main__":
    # blabla
    blabla