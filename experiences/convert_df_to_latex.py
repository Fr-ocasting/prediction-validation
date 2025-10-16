import pandas as pd
import re
import io

def parse_index_exp1_2(index_name: str) -> dict:
    """
    Parses the index string for the first and second experiments.
    It identifies the baseline, contextual data presence, and integration strategy.
    """
    if 'subway_in' in index_name or 'weather' in index_name:
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
    col_spec_str = f"@{{ {col_spec} @}}"
    latex_parts = [
        r"\begin{table}[H]",
        r"    \centering",
        fr"    \caption{{{caption}}}",
        fr"    \label{{tab:{label}}}",
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
            row_str_parts.append(f"{row[(metric, 'mean')]:.4f}")
            row_str_parts.append(f"{row[(metric, 'std')]:.4f}")
        
        latex_parts.append("        " + " & ".join(row_str_parts) + r" \\")

    latex_parts.extend([r"        \bottomrule", r"    \end{tabular}", r"\end{table}"])
    
    return "\n".join(latex_parts)
