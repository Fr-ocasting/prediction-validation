import sys
import os
import pandas as pd
import torch
import numpy as np
from datetime import datetime
import pickle 

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
NAME = 'bike_in'
FILE_BASE_NAME = 'velov'
DIRECTION = 'attracted' # attracted
FILE_PATTERN = f'velov_{DIRECTION}_by_station' # Sera complété par args.freq
DATA_SUBFOLDER = 'agg_data/velov' # Sous-dossier dans FOLDER_PATH


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
DATE_COL = 'date_retour' 
LOCATION_COL = 'id_retour' # Ou 'id_entree' pour 'attracted'?
VALUE_COL = 'volume'

def load_data(FOLDER_PATH, invalid_dates, coverage_period, args, minmaxnorm,standardize, normalize=True,
              file_pattern = FILE_PATTERN,
              data_subfolder = DATA_SUBFOLDER,
              date_col = DATE_COL,
              location_col = LOCATION_COL,
              value_col = VALUE_COL,
              direction = DIRECTION,
              name = NAME,
              file_base_name = FILE_BASE_NAME):
    """
    Charge, pivote, filtre et pré-traite les données velov (emitted).
    """
    target_freq = args.freq
    # Construction spécifique du nom de fichier pour velov
    file_name = f"{file_pattern}{target_freq}"
    data_file_path = os.path.join(FOLDER_PATH, data_subfolder, f"{file_name}.csv")

    print(f"Chargement des données depuis : {data_file_path}")
    try:
        df = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print(f"ERREUR : Le fichier {data_file_path} n'a pas été trouvé.")
        print(f"Vérifiez que la fréquence '{target_freq}' existe pour velov_{direction} et que les chemins sont corrects.")
        return None
    except Exception as e:
        print(f"ERREUR lors du chargement du fichier {file_name}.csv: {e}")
        return None

    # --- Prétraitement ---
    try:
        # Renommer pour utiliser les noms génériques (si différents)
        # df = df.rename(columns={'date_sortie': DATE_COL, 'id_sortie': LOCATION_COL, 'volume': VALUE_COL})

        # Convertir en datetime
        df[date_col] = pd.to_datetime(df[date_col])

        # Pivoter le DataFrame
        print("Pivotage du DataFrame...")
        df_pivoted = df.pivot_table(index=date_col, columns=location_col, values=value_col, aggfunc='sum')

        # Remplacer les NaN par 0
        df_pivoted = df_pivoted.fillna(0)

        # S'assurer que l'index est un DatetimeIndex
        df_pivoted.index = pd.to_datetime(df_pivoted.index)

        # Filtrage temporel basé sur l'intersection
        print(f"Filtrage temporel sur {len(coverage_period)} dates...")
        df_filtered = df_pivoted[df_pivoted.index.isin(coverage_period)].copy()

        if df_filtered.empty:
             print(f"ERREUR : Aucune donnée restante après filtrage temporel pour {file_name}.csv")
             # ... (messages d'erreur comme ci-dessus) ...
             return None

        print(f"Données filtrées. Dimensions: {df_filtered.shape}")

        # Conversion en Tensor
        data_T = torch.tensor(df_filtered.values).float()

    except KeyError as e:
        print(f"ERREUR: Colonne manquante dans {file_name}.csv : {e}. Vérifiez les noms de colonnes.")
        return None
    except Exception as e:
        print(f"ERREUR pendant le prétraitement des données {file_name}.csv: {e}")
        return None
    
    if ('agg_iris_target_n' in args.contextual_kwargs[NAME].keys()) and (args.contextual_kwargs[NAME]['agg_iris_target_n'] is not None):
        target_n = args.contextual_kwargs[NAME]['agg_iris_target_n']
        dic_path = f"{FOLDER_PATH}/dic_lyon_iris_agg{target_n}.pkl"
        dictionnary_aggregated_iris = pickle.load(open(dic_path,'rb'))
        data_T_bis = torch.empty(target_n,data_T.size(-1))

        for k,(key,list_idx) in enumerate(dictionnary_aggregated_iris.items()):
            data_T_bis[k] = torch.index_select(data_T,0,torch.tensor(list_idx).long()).mean(0) 

        ## NE PAS LOAD DE IRIS AGG MAIS PRODUIRE UN .PY ou .IPYNB QUI GENERE LE dic_lyon_iris_agg{target_n}.pkl directement adapté au BIKE_IN
        ## LANCER CE TRUC ET FAIRE L AGGREGATION BIKE IN COMME CA
        ## FAIRE ATTENTION A L ORDRE DES IRIS. 
        station_ids = iris_agg['STATION'][~iris_agg['STATION'].isna()]
        station_ids

        agg_df_pivoted = pd.DataFrame(columns = station_ids.index)
        for idx,station_id in station_ids.items():
            columns = list(map(int,station_id.split(' ')))
            effective_columns = [c for c in columns if c in df_pivoted.columns]
            agg_df_pivoted[idx] = df_pivoted[effective_columns].sum(axis=1)
        agg_df_pivoted



    mask_min = 1 
    mask = agg_df_pivoted.mean() > mask_min
    agg_df_pivoted = agg_df_pivoted.T[mask].T

    # --- Création et Prétraitement avec PersonnalInput ---
    print("Création et prétraitement de l'objet PersonnalInput...")
    dims = [0] # if [0] then Normalisation on temporal dim

    processed_input = load_input_and_preprocess(dims = dims,normalize=normalize,invalid_dates=invalid_dates,args=args,data_T=data_T,coverage_period=coverage_period,name=name,
                                                minmaxnorm=minmaxnorm,standardize=standardize)

    # --- Finalisation Métadonnées ---
    processed_input.spatial_unit = df_filtered.columns.tolist()
    processed_input.C = C
    processed_input.periods = None # Pas de périodicité spécifique définie ici

    print(f"Chargement et prétraitement de {file_base_name} terminés.")
    return processed_input

# --- Point d'entrée pour exécution directe (optionnel, pour tests) ---
# ... (Similaire à subway_indiv.py, adapter les mocks) ...