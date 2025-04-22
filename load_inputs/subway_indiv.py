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

# --- Constantes spécifiques à cette donnée ---
# Le nom de fichier sera construit dynamiquement basé sur args.freq
FILE_BASE_NAME = 'subway_indiv'
DATA_SUBFOLDER = 'validation_individuelle' # Sous-dossier dans FOLDER_PATH

# Fréquence native la plus fine disponible (pour info, le chargement dépendra de args.freq)
NATIVE_FREQ = '3min'
# Couverture théorique (à remplacer par les vraies dates si connues)
START = '2019-10-01' # Exemple basé sur head()
END = '2020-04-01'
# Liste des périodes invalides (à compléter si nécessaire)
list_of_invalid_period = []

C = 1 # Nombre de canaux/features par unité spatiale

# Colonnes attendues dans le CSV
DATE_COL = 'VAL_DATE'
LOCATION_COL = 'COD_TRG'
VALUE_COL = 'Flow'

def load_data(dataset, ROOT, FOLDER_PATH, invalid_dates, intersect_coverage_period, args, normalize=True):
    """
    Charge, pivote, filtre et pré-traite les données subway_indiv.

    Args:
        dataset (DataSet): L'objet DataSet principal (pour contexte).
        ROOT (str): Chemin racine du projet ou des données.
        FOLDER_PATH (str): Chemin vers le dossier contenant DATA_SUBFOLDER.
        invalid_dates (list): Liste des périodes invalides fournie globalement.
        intersect_coverage_period (pd.DatetimeIndex): Période temporelle à conserver.
        args (Namespace): Arguments globaux (contient args.freq).
        normalize (bool): Faut-il normaliser les données.

    Returns:
        PersonnalInput: Objet contenant les données traitées.
    """
    target_freq = args.freq
    file_name = f"{FILE_BASE_NAME}_{target_freq}"
    data_file_path = os.path.join(ROOT, FOLDER_PATH, DATA_SUBFOLDER, file_name, f"{file_name}.csv")

    print(f"Chargement des données depuis : {data_file_path}")
    try:
        df = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print(f"ERREUR : Le fichier {data_file_path} n'a pas été trouvé.")
        print(f"Vérifiez que la fréquence '{target_freq}' existe pour {FILE_BASE_NAME} et que les chemins sont corrects.")
        return None
    except Exception as e:
        print(f"ERREUR lors du chargement du fichier {file_name}.csv: {e}")
        return None

    # --- Prétraitement ---
    try:
        # Convertir en datetime
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

        # Pivoter le DataFrame
        print("Pivotage du DataFrame...")
        df_pivoted = df.pivot_table(index=DATE_COL, columns=LOCATION_COL, values=VALUE_COL, aggfunc='sum')

        # Remplacer les NaN (dus au pivot) par 0
        df_pivoted = df_pivoted.fillna(0)

        # S'assurer que l'index est un DatetimeIndex
        df_pivoted.index = pd.to_datetime(df_pivoted.index)

        # Ré-échantillonage (si args.freq est différent de la freq native du fichier ET plus grossier - rare ici car on charge le fichier exact)
        # Note: Normalement, on charge le fichier correspondant à args.freq, donc pas besoin de resample ici.
        # Laissez ce bloc commenté sauf si vous avez besoin de charger une fréquence plus fine et de la ré-échantillonner.
        # current_freq = pd.infer_freq(df_pivoted.index) # Ou extraire de NATIVE_FREQ/target_freq
        # if target_freq != current_freq and pd.to_timedelta(target_freq) > pd.to_timedelta(current_freq):
        #    print(f"Ré-échantillonage de {current_freq} vers {target_freq}...")
        #    df_pivoted = df_pivoted.resample(target_freq).sum() # ou .mean() selon la donnée

        # Filtrage temporel basé sur l'intersection
        print(f"Filtrage temporel sur {len(intersect_coverage_period)} dates...")
        # Assurez-vous que les deux index sont bien des DatetimeIndex
        df_filtered = df_pivoted[df_pivoted.index.isin(intersect_coverage_period)].copy()
        local_df_dates = pd.DataFrame(df_filtered.index, columns=['date'])

        if df_filtered.empty:
             print(f"ERREUR : Aucune donnée restante après filtrage temporel pour {file_name}.csv")
             print(f"Vérifiez la couverture de intersect_coverage_period ({intersect_coverage_period.min()} - {intersect_coverage_period.max()})")
             print(f"et la couverture du fichier chargé ({df_pivoted.index.min()} - {df_pivoted.index.max()})")
             return None

        print(f"Données filtrées. Dimensions: {df_filtered.shape}")

        # Conversion en Tensor
        data_T = torch.tensor(df_filtered.values).float()

    except KeyError as e:
        print(f"ERREUR: Colonne manquante dans {file_name}.csv : {e}. Vérifiez DATE_COL, LOCATION_COL, VALUE_COL.")
        return None
    except Exception as e:
        print(f"ERREUR pendant le prétraitement des données {file_name}.csv: {e}")
        return None


    # --- Création et Prétraitement avec PersonnalInput ---
    # Utilisation de la fonction helper locale
    print("Création et prétraitement de l'objet PersonnalInput...")
    processed_input = load_input_and_preprocess(
        dims=[0], # Normalisation sur la dimension temporelle par défaut
        normalize=normalize,
        invalid_dates=invalid_dates, # Utilise les invalid_dates globales passées
        args=args,
        data_T=data_T,
        dataset=dataset, # Passe le dataset principal pour contexte
        df_dates=local_df_dates
    )

    if processed_input is None:
        return None

    # --- Finalisation Métadonnées ---
    processed_input.spatial_unit = df_filtered.columns.tolist()
    processed_input.C = C
    processed_input.periods = None # Pas de périodicité spécifique définie ici

    print(f"Chargement et prétraitement de {FILE_BASE_NAME} terminés.")
    return processed_input

def load_input_and_preprocess(dims, normalize, invalid_dates, args, data_T, dataset, df_dates):
    """
    Fonction helper pour instancier PersonnalInput à partir d'un Tensor
    et appeler preprocess.
    """
    # Filtrer les arguments de args qui sont pertinents pour DataSet/PersonnalInput
    args_DataSet = filter_args(DataSet, args)

    try:
        # Instancier PersonnalInput avec le Tensor
        # Note: on passe le time_step_per_hour et potentiellement d'autres
        # attributs de contexte depuis le 'dataset' principal.
        personal_instance = PersonnalInput(
            invalid_dates, # Les dates invalides spécifiques à cette donnée (ou globales)
            args,          # L'objet args complet
            tensor=data_T,
            dates=df_dates,
            time_step_per_hour=getattr(dataset, 'time_step_per_hour', None), # Hérite du dataset principal
            dims=dims,     # Dimensions pour la normalisation éventuelle
            # minmaxnorm=getattr(dataset, 'minmaxnorm', None),   # Hérite si besoin
            # standardize=getattr(dataset, 'standardize', None), # Hérite si besoin
            **args_DataSet # Arguments filtrés de args pour DataSet
        )

        # Appeler la méthode preprocess de l'instance
        print("Appel de la méthode preprocess...")
        personal_instance.preprocess(
            args.train_prop,
            args.valid_prop,
            args.test_prop,
            args.train_valid_test_split_method,
            normalize=normalize
        )
        print("Méthode preprocess terminée.")
        return personal_instance

    except Exception as e:
        print(f"ERREUR lors de l'instanciation ou preprocess de PersonnalInput : {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # blabla
    blabla