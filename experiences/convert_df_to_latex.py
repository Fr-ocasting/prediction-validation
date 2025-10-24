import pandas as pd
import re
import io

def parse_index_exp1_2(index_name: str,contextual= 'subway_in') -> dict:
    """
    Parses the index string for the first and second experiments.
    It identifies the baseline, contextual data presence, and integration strategy.
    """
    if contextual in index_name or 'weather' in index_name:
        contextual_data = 'Yes'
        strategy_match = re.search(r'calendar_(.*?)__e', index_name)
        strategy = strategy_match.group(1).replace('_', ' ').replace(' t ', '-t-').replace(' s ', '-s-').title()
    else:
        contextual_data = 'No'
        strategy = 'Baseline (No Context)'
        
    return {
        "Contextual Data": contextual_data,
        "Integration Strategy": strategy,
    }

def parse_index_exp4(index_name: str) -> dict:
    """
    Parses the index string for the heterogeneous data experiment.
    It identifies the baseline and extracts CrossAttnBackBone hyperparameters.
    """
    if 'subway_in_subway_out' not in index_name:
        return {
            "Type": "Baseline",
            "InEmb": "N/A", "ctxInEmb": "N/A", "adp": "N/A", "adpQ": "N/A",
        }
    else:
        # Helper to find a parameter's value, defaulting to '0' if not present
        def find_param(param, name):
            p_match = re.search(fr'{param}(\d+)', name)
            return p_match.group(1) if p_match else '0'

        return {
            "Type": "CrossAttnBackBone",
            "InEmb": find_param('InEmb', index_name),
            "ctxInEmb": find_param('ctxInEmb', index_name),
            "adp": find_param('adp', index_name),
            "adpQ": find_param('adpQ', index_name),
        }

def dataframe_to_latex(df: pd.DataFrame, caption: str, label: str, index_parser: callable) -> str:
    """
    Converts a pandas DataFrame with a multi-level column into a formatted LaTeX table.

    Args:
        df (pd.DataFrame): The input DataFrame.
        caption (str): The caption for the table.
        label (str): The LaTeX label for the table.
        index_parser (callable): A function to parse the DataFrame index into new columns.
    
    Returns:
        str: A string containing the full LaTeX code for the table.
    """
    df.columns = df.columns.remove_unused_levels()
    # --- 1. Process Index and Augment DataFrame ---
    parsed_index_data = df.index.map(index_parser).to_list()
    df_parsed = pd.DataFrame(parsed_index_data, index=df.index)
    df_full = pd.concat([df_parsed, df], axis=1)

    # Sort to ensure baseline is last for visual separation
    sort_key = df_full['Contextual Data'] == 'No' if 'Contextual Data' in df_full.columns else df_full['Type'] == 'Baseline'
    df_full = df_full.iloc[sort_key.argsort(kind='mergesort')]

    # --- 2. Prepare Headers ---
    metrics = df.columns.levels[0].to_list()
    horizon = re.search(r'h(\d+)', metrics[0]).group(1)
    metric_names = [re.sub(r'_\s*h\d+\s*', '', m).replace('_', ' ').upper() for m in metrics]
    
    custom_cols = df_parsed.columns.to_list()
    num_custom_cols = len(custom_cols)
    num_metric_cols = 2 * len(metric_names)
    col_spec = 'l' * num_custom_cols + 'c' * num_metric_cols

    # --- 3. Build LaTeX String ---
    col_spec_str = f"{{{col_spec}}}"
    latex_parts = [
        r"\begin{table}[!htb]",
        r"    \centering",
        fr"    \caption{{{caption}}}",
        fr"    \label{{tab:{label}}}",
        r"    \resizebox{\textwidth}{!}{",
        fr"    \begin{{tabular}}{{{col_spec_str}}}",
        r"        \toprule"
    ]

    # Header Row 1 (Metric Names)
    header1_parts = [' & '.join(custom_cols)]
    header1_parts.extend([fr"\multicolumn{{2}}{{c}}{{ {name} (h{horizon}) }}" for name in metric_names])
    latex_parts.append("        " + ' & '.join(header1_parts) + r" \\")

    # Header Row 2 (cmidrule)
    cmidrules = [fr"\cmidrule(lr){{1-{num_custom_cols}}}"]
    for i in range(len(metric_names)):
        start_col = num_custom_cols + 2 * i + 1
        end_col = start_col + 1
        cmidrules.append(fr"\cmidrule(lr){{{start_col}-{end_col}}}")
    latex_parts.append("        " + ' '.join(cmidrules))

    # Header Row 3 (Mean/Std)
    header3_parts = [''] * num_custom_cols + [r"Mean & Std"] * len(metric_names)
    latex_parts.append("        " + ' & '.join(header3_parts) + r" \\")
    latex_parts.append(r"        \midrule")

    # --- 4. Add Data Rows ---
    for index, row in df_full.iterrows():
        if row.get("Contextual Data") == 'No' or row.get("Type") == 'Baseline':
            latex_parts.append(r"        \midrule")
        
        row_str_parts = [str(row[c]) for c in custom_cols]
        for metric in metrics:
            row_str_parts.append(f"{row[(metric, 'mean')]}")
            row_str_parts.append(f"{row[(metric, 'std')]:.4f}")
        
        latex_parts.append("        " + " & ".join(row_str_parts) + r" \\")

    latex_parts.extend([r"        \bottomrule", r"    \end{tabular}", r"}",r"\end{table}"])
    
    return "\n".join(latex_parts)



def load_csv(folder_path,dic_exp_to_names,exp_i,trial_j,n_bis,df_j_all,metric_i,metrics):
    local_folder_path =  f"{folder_path}/{exp_i}/{dic_exp_to_names[exp_i]}/{trial_j}_bis{n_bis}"
    Losses_file_path = f"{local_folder_path}/Losses_{trial_j}_bis{n_bis}.csv"
    df_j = pd.read_csv(Losses_file_path,index_col = 0)
    df_j.columns = [f'Train_{n_bis}',f'Valid_{n_bis}']
    df_j_all = pd.concat([df_j_all,df_j],axis=1)

    # Get metrics : 
    metrics_ij = pd.read_csv(f"{local_folder_path}/METRICS_{trial_j}_bis{n_bis}.csv",index_col = 0)
    re._pattern = re.compile(r'_h(\d+)$')
    horizons = list(set([re._pattern.findall(c)[0] for c in metrics_ij.columns if len(re._pattern.findall(c)) > 0]))
    if len(horizons) > 1:
        raise ValueError(f"Multiple horizons found in columns_metrics : {horizons}")
    else:
        horizon = horizons[0]
        columns_metrics = [f"{m}_h{horizon}" for m in metrics]
    if set(columns_metrics).issubset(metrics_ij.columns) :
        metrics_ij = metrics_ij.loc['test',columns_metrics].copy()
    else:
        metrics_ij = None

    if metrics_ij is not None: 
        metric_i.append(metrics_ij)
    return df_j_all, metric_i

def tackle_trial_j(folder_path,dic_exp_to_names,L_metrics,exp_i,trial_j,metrics):
    df_j_all = pd.DataFrame()
    metric_i = []
    for n_bis in range(1,6):

        df_j_all, metric_i = load_csv(folder_path,dic_exp_to_names,exp_i,trial_j,n_bis,df_j_all,metric_i,metrics)
    
    if len(metric_i) > 0: 
        metric_i = pd.DataFrame(metric_i)
        metric_i.index = [f"{trial_j}_bis{n_bis}" for n_bis in range(1,6)]
        L_metrics.append(metric_i)

    return L_metrics


def build_legend_group_exp4(x):
    if ('adpQ0' in x) and ('adp0' in x):
        return 'adp0 & adpQ0'
    if ('adpQ0' in x):
        return 'adpQ0'
    if ('adp0' in x):
        return 'adp0'
    if not('adp' in x):
        return 'Baseline'
    return 'STAEformer_CrossAttn'

def update_df_metrics_exp4_15min(df_metrics_all):
    df_metrics_all['legend_group'] = df_metrics_all.reset_index()['index'].apply(build_legend_group_exp4).values
    df_metrics_all['id'] = [c.split('_CrossAttnBackBone_')[1].split('__')[0] if '_CrossAttnBackBone_' in c else 'Baseline' for c in df_metrics_all.index]
    df_metrics_all['id'] = df_metrics_all['id'].fillna('Baseline')
    df_metrics_all = df_metrics_all.rename(columns= {'rmse_h4':'rmse','rmse_h1':'rmse','mae_h4':'mae','mae_h1':'mae','mase_h4':'mase','mase_h1':'mase'})
    return df_metrics_all

def update_df_metrics_Exp6_subway_netmob(df_metrics_all):
    df_metrics_all['legend_group'] = df_metrics_all.reset_index()['index'].apply(build_legend_group_exp4).values
    df_metrics_all['id'] = [c.split('_CrossAttnBackBone_')[1].split('__')[0] if '_CrossAttnBackBone_' in c else 'Baseline' for c in df_metrics_all.index]
    df_metrics_all['id'] = df_metrics_all['id'].fillna('Baseline')
    df_metrics_all = df_metrics_all.rename(columns= {'rmse_h4':'rmse','rmse_h1':'rmse','mae_h4':'mae','mae_h1':'mae','mase_h4':'mase','mase_h1':'mase'})
    return df_metrics_all    

def build_legend_group_exp1(x):
    if not('subway_in_subway_out' in x) and not('subway_out_subway_in' in x):
        return 'Baseline'
    elif 'independant_embedding' in x:
        return 'Independant Embedding'
    elif 'shared_embedding' in x:
        return 'Shared Embedding'
    else:
        return 'Other Methods'

def update_df_metrics_exp1(df_metrics_all,target_data='subway_in'):
    df_metrics_all['legend_group'] = df_metrics_all.reset_index()['index'].apply(build_legend_group_exp1).values
    df_metrics_all['id'] = [c.split('_calendar_')[1].split('__')[0] if (('subway_in_subway_out' in c) or ('subway_out_subway_in' in c)) else 'Baseline' for c in df_metrics_all.index]
    df_metrics_all = df_metrics_all.rename(columns= {'rmse_h4':'rmse','rmse_h1':'rmse','mae_h4':'mae','mae_h1':'mae','mase_h4':'mase','mase_h1':'mase'})

    return df_metrics_all

def update_df_metrics(df_metrics_all,exp_i):
    if exp_i == 'Exp1':
        df =  update_df_metrics_exp1(df_metrics_all)
    elif exp_i == 'Exp1_subway_in':
        df =  update_df_metrics_exp1(df_metrics_all,'subway_in')
    elif exp_i == 'Exp1_subway_out':
        df =  update_df_metrics_exp1(df_metrics_all,'subway_out')
    elif exp_i == 'Exp4_15min':
        df =  update_df_metrics_exp4_15min(df_metrics_all)
    elif exp_i == 'Exp6_subway_netmob':
       df = update_df_metrics_Exp6_subway_netmob(df_metrics_all)
    else:
        raise NotImplementedError

    df.id = df.id.apply(lambda x: x.replace('adp_query_cross_attn_traffic_model_backbone','CABB'))
    return df 