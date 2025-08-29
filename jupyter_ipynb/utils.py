
import io 
import pandas as pd 
import re 

def parse_results_to_dataframe(data_string, bis = False):
    """
    Analyse une chaîne de caractères de résultats de modèles pour la convertir en DataFrame Pandas.
    """
    # Utilise io.StringIO pour lire la chaîne de caractères multi-lignes comme un fichier.
    lines = io.StringIO(data_string).readlines()
    
    parsed_data = []

    # Correspondance entre l'horizon et le temps
    horizon_map = {
        'h1': '+15min',
        'h2': '+30min',
        'h3': '+45min',
        'h4': '+60min', # Correction du h5 de la demande en h4 présent dans les données
    }

    # Expression régulière pour extraire les informations de chaque ligne
    # Elle capture le nom de l'expérience et les valeurs des métriques.
    line_regex = re.compile(r"(_\w+):.*?RMSE = ([\d.]+), MAE = ([\d.]+), MAPE = ([\d.]+)")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = line_regex.search(line)

        if not match:
            continue
            
        # Extrait les parties de la ligne
        description, rmse, mae, mape = match.groups()
        parts = description.strip('_').split('_')

        # Identifie la donnée cible (target) et les données contextuelles
        target_data = f"{parts[0]}_{parts[1]}"
        if bis == False:
            context_parts = '_'.join(parts[2:]) # Ignore target, calendar, embedding, et horizon
            horizon_code = parts[-1]
            bis_code = [1]*len(horizon_code)
        else:
            context_parts = '_'.join(parts[2:]) # Ignore target, calendar, embedding, et horizon
            horizon_code = parts[-2]
            bis_code = parts[-1][3:]
        
        
        horizon = horizon_map.get(horizon_code, 'N/A')

        # Crée un dictionnaire pour stocker les informations de la ligne
        row = {
            'target_data': target_data,
            'subway_in': 'subway_in' in context_parts,
            'subway_out': 'subway_out' in context_parts,
            'bike_in': 'bike_in' in context_parts,
            'bike_out': 'bike_out' in context_parts,
            'weather': 'weather' in context_parts,
            'epochs': parts[-3] if (parts[-3].startswith('e') and parts[-2].startswith('h') and not parts[-3].startswith('em'))  else '',
            'horizon': horizon,
            'horizon_code': horizon_code, # Ajout pour le tri
            'bis': bis_code,
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'stack': 'stack' in line,
            'ff_concat_late': 'ff_concat_late' in line,
            'attn_late': 'attn_late' in line,
            'STAEformer': 'STAEformer' in line
        }
        parsed_data.append(row)

    if not parsed_data:
        return pd.DataFrame()

    # Crée le DataFrame
    df = pd.DataFrame(parsed_data)
    
    # Trie les valeurs comme demandé : par cible, puis par horizon
    df = df.sort_values(by=['target_data', 'horizon_code','bis']).reset_index(drop=True)
    
    # Réorganise et supprime les colonnes non nécessaires pour l'affichage final
    final_columns = [
        'target_data', 'horizon', 
        'subway_in', 'subway_out', 
        'bike_in', 'bike_out',
        'RMSE', 'MAE', 'MAPE','bis',
        'stack','ff_concat_late', 'STAEformer','attn_late','epochs','weather'
    ]
    df = df[final_columns]
    
    return df

def display_latex_df(results_df):
    results_df.columns = ['\_'.join(c.split('_')) for c in results_df.columns]
    for metric in ['RMSE','MAE','MAPE']:
        results_df[metric] =  results_df[metric].apply(lambda x : '{:.2f}'.format(x))
    df_subway_in = results_df[results_df['target\_data'] == 'subway_in'].copy()
    df_subway_out = results_df[results_df['target\_data'] == 'subway_out'].copy()
    for col in df_subway_in.columns:
        df_subway_in[col] = df_subway_in[col].replace({True: '\\checkmark', False: ''})
    for col in df_subway_out.columns:
        df_subway_out[col] = df_subway_out[col].replace({True: '\\checkmark', False: ''})

    df_subway_in=df_subway_in.drop(columns = ['target\_data','subway\_in']).set_index('horizon')
    df_subway_out=df_subway_out.drop(columns = ['target\_data','subway\_out']).set_index('horizon')

    # Affiche les DataFrames séparés par target_data pour une meilleure lisibilité
    print("%--- DataFrame pour Target Data: subway_in ---")
    print('\\begin{frame}')
    print('    \\begin{table}[htbp]')
    print('        \\centering')
    print('        \\tiny')
    print("        \\caption{Résultats des prédictions pour la cible 'subway\_in'}")
    print("        \\label{tab:results_subway_in}")
    # print("        \\begin{tabular}{cccc|ccc}")
    print(df_subway_in.to_latex())
    print("    \end{table}")
    print("\end{frame}")


    print("\n\n%--- DataFrame pour Target Data: subway_out ---")
    print('\\begin{frame}')
    print('    \\begin{table}[htbp]')
    print('        \\centering')
    print('        \\tiny')
    print("        \\caption{Résultats des prédictions pour la cible 'subway\_out'}")
    print("        \\label{tab:results_subway_out}")
    print(df_subway_out.to_latex())
    print("    \end{table}")
    print("\end{frame}")

if __name__ == '__main__':
    # Exemple de code pour utiliser ces fonctions :
        results_string = """ _subway_out_subway_in_calendar_embedding_h1: All Steps RMSE = 32.35, MAE = 18.14, MAPE = 22.47, MSE = 1046.60
                             _subway_out_subway_in_bike_in_calendar_embedding_h1: All Steps RMSE = 31.56, MAE = 18.05, MAPE = 21.73, MSE = 995.97
                             """
        # Exécute la fonction et affiche le DataFrame
        results_df = parse_results_to_dataframe(results_string,bis= False)
        # display(results_df) 
        display_latex_df(results_df)
        
        results_string = """_subway_out_subway_in_calendar_embedding_h1_bis1: All Steps RMSE = 32.35, MAE = 18.14, MAPE = 22.47, MSE = 1046.60
                            _subway_out_subway_in_bike_in_calendar_embedding_h1_bis1: All Steps RMSE = 31.56, MAE = 18.05, MAPE = 21.73, MSE = 995.97
                            """
        # Exécute la fonction et affiche le DataFrame
        results_df = parse_results_to_dataframe(results_string,bis= True)
        # display(results_df) 
        display_latex_df(results_df)


