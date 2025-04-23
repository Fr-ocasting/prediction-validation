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
FILE_BASE_NAME = 'velov'
DIRECTION = 'emitted'
FILE_PATTERN = f'velov_{DIRECTION}_by_station' # Sera complété par args.freq
DATA_SUBFOLDER = 'agg_data/velov' # Sous-dossier dans FOLDER_PATH

# Fréquence native la plus fine disponible (pour info)
NATIVE_FREQ = '2min' # A vérifier, basé sur votre commentaire
# Couverture théorique
START = '2019-01-01' # Exemple basé sur head()
END = '2020-01-01'
# Liste des périodes invalides
list_of_invalid_period = []

C = 1 # Nombre de canaux/features par unité spatiale

# Colonnes attendues dans le CSV
DATE_COL = 'date_sortie' # Ou 'date_entree' pour 'attracted'?
LOCATION_COL = 'id_sortie' # Ou 'id_entree' pour 'attracted'?
VALUE_COL = 'volume'

def load_data(ROOT, FOLDER_PATH, invalid_dates, coverage_period, args, normalize=True):
    """
    Charge, pivote, filtre et pré-traite les données velov (emitted).
    """
    target_freq = args.freq
    # Construction spécifique du nom de fichier pour velov
    file_name = f"{FILE_PATTERN}{target_freq}"
    data_file_path = os.path.join(ROOT, FOLDER_PATH, DATA_SUBFOLDER, f"{file_name}.csv")

    print(f"Chargement des données depuis : {data_file_path}")
    try:
        df = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print(f"ERREUR : Le fichier {data_file_path} n'a pas été trouvé.")
        print(f"Vérifiez que la fréquence '{target_freq}' existe pour velov_{DIRECTION} et que les chemins sont corrects.")
        return None
    except Exception as e:
        print(f"ERREUR lors du chargement du fichier {file_name}.csv: {e}")
        return None

    # --- Prétraitement ---
    try:
        # Renommer pour utiliser les noms génériques (si différents)
        # df = df.rename(columns={'date_sortie': DATE_COL, 'id_sortie': LOCATION_COL, 'volume': VALUE_COL})

        # Convertir en datetime
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

        # Pivoter le DataFrame
        print("Pivotage du DataFrame...")
        df_pivoted = df.pivot_table(index=DATE_COL, columns=LOCATION_COL, values=VALUE_COL, aggfunc='sum')

        # Remplacer les NaN par 0
        df_pivoted = df_pivoted.fillna(0)

        # S'assurer que l'index est un DatetimeIndex
        df_pivoted.index = pd.to_datetime(df_pivoted.index)

        # Filtrage temporel basé sur l'intersection
        print(f"Filtrage temporel sur {len(coverage_period)} dates...")
        df_filtered = df_pivoted[df_pivoted.index.isin(coverage_period)].copy()
        local_df_dates = pd.DataFrame(df_filtered.index, columns=['date'])

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

    # --- Création et Prétraitement avec PersonnalInput ---
    print("Création et prétraitement de l'objet PersonnalInput...")
    dims = [0] # if [0] then Normalisation on temporal dim

    processed_input = load_input_and_preprocess(dims = dims,normalize=normalize,invalid_dates=invalid_dates,args=args,data_T=data_T,coverage_period=coverage_period)

    # --- Finalisation Métadonnées ---
    processed_input.spatial_unit = df_filtered.columns.tolist()
    processed_input.C = C
    processed_input.periods = None # Pas de périodicité spécifique définie ici

    print(f"Chargement et prétraitement de {FILE_BASE_NAME} terminés.")
    return processed_input

# --- Point d'entrée pour exécution directe (optionnel, pour tests) ---
# ... (Similaire à subway_indiv.py, adapter les mocks) ...