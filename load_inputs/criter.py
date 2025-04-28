import sys
import os
import pandas as pd
import torch
import numpy as np
from datetime import datetime
import glob # Pour lister les fichiers

# --- Gestion de l'arborescence ---
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- Importations personnalisées ---
from dataset import DataSet, PersonnalInput
from build_inputs.load_preprocessed_dataset import load_input_and_preprocess
from utils.utilities import filter_args # Assurez-vous que ce chemin est correct
# Import de votre fonction spécifique pour CRITER
try:
    # Ajustez le chemin si nécessaire
    from build_inputs.load_raw_data import load_CRITER
except ImportError:
    print("ERREUR: Impossible d'importer 'load_CRITER'. Vérifiez le chemin dans 'load_inputs/criter.py'")
    # Définir une fonction factice pour éviter les erreurs si l'import échoue
    def load_CRITER(file_path):
        print(f"AVERTISSEMENT: load_CRITER non trouvé, chargement de {file_path} échouera.")
        # Retourne un DataFrame vide avec les colonnes attendues pour éviter des erreurs aval
        return pd.DataFrame(columns=['HORODATE', 'ID_POINT_MESURE', 'DEBIT_HEURE'])


# --- Constantes spécifiques à cette donnée ---
FILE_BASE_NAME = 'CRITER'
DATA_SUBFOLDER_PATTERN = 'Comptages_Velo_Routier/CRITER/6 min {year}' # Sera formaté avec l'année

NATIVE_FREQ = '6min' # La fréquence des fichiers bruts
# Couverture théorique
START = '2019-01-01' # Exemple basé sur head()
END = '2020-01-01'
# Liste des périodes invalides
list_of_invalid_period = []

C = 1 # Nombre de canaux/features par unité spatiale

# Colonnes attendues DANS LE RETOUR de load_CRITER (à adapter si load_CRITER retourne autre chose)
DATE_COL = 'HORODATE'
LOCATION_COL = 'ID_POINT_MESURE'
VALUE_COL = 'DEBIT_HEURE' # Ou une autre colonne pertinente retournée par load_CRITER

def load_data(FOLDER_PATH, invalid_dates, coverage_period, args, normalize=True):
    """
    Charge, concatène, pivote, filtre et pré-traite les données CRITER.
    """
    target_freq = args.freq
    # CRITER a une fréquence native de 6min. On chargera ces fichiers puis on resamplera.
    native_freq_criter = NATIVE_FREQ

    # Déterminer les années couvertes par coverage_period
    min_date = coverage_period.min()
    max_date = coverage_period.max()
    years_to_load = range(min_date.year, max_date.year + 1)

    all_files_to_load = []
    for year in years_to_load:
        criter_path_year = os.path.join(FOLDER_PATH, DATA_SUBFOLDER_PATTERN.format(year=year))
        # Lister tous les fichiers .txt dans le dossier de l'année
        # Attention: ceci charge tous les fichiers de l'année, même ceux hors période.
        # Un filtrage plus fin sur les noms de fichiers pourrait être nécessaire si beaucoup de fichiers.
        files_in_year = sorted(glob.glob(os.path.join(criter_path_year, "*.txt")))
        if not files_in_year:
             print(f"AVERTISSEMENT: Aucun fichier CRITER trouvé pour l'année {year} dans {criter_path_year}")
        all_files_to_load.extend(files_in_year)

    if not all_files_to_load:
        print(f"ERREUR: Aucun fichier CRITER trouvé pour les années {list(years_to_load)} dans {FOLDER_PATH}")
        return None

    print(f"Chargement de {len(all_files_to_load)} fichiers CRITER...")
    list_df_criter = []
    for f_path in all_files_to_load:
        try:
            df_temp = load_CRITER(f_path)
            list_df_criter.append(df_temp)
        except Exception as e:
            print(f"ERREUR lors du chargement du fichier CRITER {f_path} avec load_CRITER : {e}")
            # Optionnel : décider si on continue sans ce fichier ou si on arrête tout

    if not list_df_criter:
        print("ERREUR: Aucun DataFrame CRITER n'a pu être chargé.")
        return None

    # Concaténer tous les DataFrames chargés
    print("Concaténation des DataFrames CRITER...")
    df = pd.concat(list_df_criter, ignore_index=True)

    # --- Prétraitement ---
    try:
        # S'assurer que la colonne date est bien en datetime
        # (load_CRITER devrait idéalement déjà le faire)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

        # Garder uniquement les colonnes nécessaires pour le pivot
        df = df[[DATE_COL, LOCATION_COL, VALUE_COL]].copy()
        
        # Supprimer les doublons potentiels (même point, même horodate) avant pivot
        df = df.drop_duplicates(subset=[DATE_COL, LOCATION_COL], keep='first')


        # Pivoter le DataFrame
        print("Pivotage du DataFrame CRITER...")
        df_pivoted = df.pivot_table(index=DATE_COL, columns=LOCATION_COL, values=VALUE_COL, aggfunc='sum') # aggfunc='sum' ou 'mean'?

        # Remplacer les NaN par 0
        df_pivoted = df_pivoted.fillna(0)

        # S'assurer que l'index est un DatetimeIndex
        df_pivoted.index = pd.to_datetime(df_pivoted.index)

        # Ré-échantillonage si target_freq est différent de 6min
        if target_freq != native_freq_criter:
            print(f"Ré-échantillonage de {native_freq_criter} vers {target_freq}...")
            try:
                 # Vérifier si la fréquence cible est plus grossière
                 if pd.to_timedelta(target_freq) >= pd.to_timedelta(native_freq_criter):
                     df_pivoted = df_pivoted.resample(target_freq).sum() # 'sum' car ce sont des débits/comptes?
                 else:
                     print(f"AVERTISSEMENT: La fréquence cible {target_freq} est plus fine que la fréquence native {native_freq_criter}. Aucun ré-échantillonage effectué.")
            except ValueError as e:
                 print(f"ERREUR lors de la tentative de ré-échantillonage vers {target_freq}: {e}")
                 # Continuer avec la fréquence native ? Ou retourner une erreur ?
                 # return None # Option plus sûre

        # Filtrage temporel basé sur l'intersection
        print(f"Filtrage temporel CRITER sur {len(coverage_period)} dates...")
        df_filtered = df_pivoted[df_pivoted.index.isin(coverage_period)].copy()
        local_df_dates = pd.DataFrame(df_filtered.index, columns=['date'])

        if df_filtered.empty:
             print(f"ERREUR : Aucune donnée CRITER restante après filtrage temporel.")
             # ... (messages d'erreur comme ci-dessus) ...
             return None

        print(f"Données CRITER filtrées. Dimensions: {df_filtered.shape}")

        # Conversion en Tensor
        data_T = torch.tensor(df_filtered.values).float()

    except KeyError as e:
        print(f"ERREUR: Colonne manquante dans les données CRITER retournées par load_CRITER : {e}.")
        return None
    except Exception as e:
        print(f"ERREUR pendant le prétraitement des données CRITER : {e}")
        return None

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
# ... (Similaire à subway_indiv.py, adapter les mocks, notamment pour ROOT et FOLDER_PATH pointant vers les données CRITER) ...