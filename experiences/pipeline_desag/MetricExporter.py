import re 
import os 
import pandas as pd
import sys

class MetricExporter:
    def __init__(self, results_str, contextual_dataset_names, known_targets=None):
        self.metrics = ['RMSE', 'MAE', 'MASE', 'MAPE']
        self.targets = known_targets or ['subway_in', 'subway_out', 'bike_out', 'bike_in', 'PeMS']
        self.contextual_dataset_names = contextual_dataset_names
        self.df = self._process_data(results_str)

    def _parse_id(self, full_id):
        # Séparation paramètres entrainement / config
        config_part = full_id.split('__')[0]
        model = config_part.split('_')[0]
        
        # Identification Target
        target = next((t for t in self.targets if config_part.startswith(f"{model}_{t}")), "unknown")

        if len(self.contextual_dataset_names) == 0:
            ctx = ""
        elif len(self.contextual_dataset_names) == 1:
            ctx = self.contextual_dataset_names[0]
        else:
            raise NotImplementedError("Multiple contextual datasets parsing not implemented.")
        
        # Extraction Contexte et Horizon
        raw_context = config_part.replace(f"{model}_{target}_", "")
        # clean_ctx = raw_context.replace('calendar', '').replace('_', ' ').strip().title()
        # display_ctx = clean_ctx if clean_ctx else "Baseline"
        # ============ modification ==========
        name_map = {
            'late_fusion': 'L_',
            'early_fusion': 'E_',
            'CrossAttnBackBone': 'CABB',
            'BackBone': 'BB',
            's_proj_t_proj': 'S-proj T-proj'
        }
        for full, short in name_map.items():
            raw_context = raw_context.replace(full, short)
        
        clean_ctx = raw_context.replace('_', ' ').strip()
        if clean_ctx and clean_ctx != "":
            if clean_ctx == 'calendar':
                cli_id = 'Baseline'
            else:
                cli_id = clean_ctx  
        else:
            cli_id = "Baseline"
        # ============ ========== ==========

        horizon_match = re.search(r'_h(\d+)_', full_id)
        horizon = horizon_match.group(1) if horizon_match else "1"

        
        # ID unique sans le suffixe 'bis' pour le groupage
        unique_config_id = f"{model}_{target}_{cli_id}_h{horizon}"
        
        return model, target, cli_id, horizon, unique_config_id,ctx

    def _process_data(self, results_str):
        data = []
        pattern = re.compile(r"([\w\_]+):\s+All Steps " + ", ".join([f"{m} = ([\d.]+)" for m in self.metrics]))
        
        for line in results_str.strip().split('\n'):
            match = pattern.search(line)
            if not match: continue
            
            groups = match.groups()
            full_id = groups[0]
            model, target, cli_id, hor, config_id, ctx = self._parse_id(full_id)
            
            row = {
                'config_id': config_id, 'Model': model, 'Target': target, 
                'Id': cli_id, 'Horizon': hor, 'Context': ctx
            }
            row.update({self.metrics[i]: float(groups[i+1]) for i in range(len(self.metrics))})
            data.append(row)
            
        df = pd.DataFrame(data)
        if df.empty: return df
        
        # Moyenne des essais 'bis' par configuration unique
        # numeric_cols = self.metrics
        # df = df.groupby(['config_id', 'Model', 'Target', 'Context', 'Horizon'])[numeric_cols].mean().reset_index()
        # ============ modification ==========
        agg_dict = {m: ['mean', 'std'] for m in self.metrics}
        df_grouped = df.groupby(['config_id', 'Model', 'Target', 'Id','Context','Horizon']).agg(agg_dict).reset_index()
        df_grouped.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_grouped.columns]
        # ============ ========== ==========
        return df_grouped

    def export_all(self, folder_path, exp_i):
        # Un tableau par Horizon et par Target
        for (target, horizon,context), group in self.df.groupby(['Target', 'Horizon','Context']):
            self._generate_table(group, folder_path, f"{target}_{context}_h{horizon}")


    def _generate_table(self, df_group, folder_path, exp_id):
        df = df_group.copy()
        print(df)
        # --- MODIFICATION : Calcul améliorations et Tri ---
        baseline_mask = df['Id'] == "Baseline"
        baseline_row = df[baseline_mask]
        others = df[~baseline_mask].copy()
        
        if not baseline_row.empty:
            print('baseline not empty')
            for m in self.metrics:
                base_val = baseline_row.iloc[0][f"{m}_mean"]
                # Gain = (Base - Trial) / Base * 100
                others[f'{m}_imp'] = ((base_val - others[f"{m}_mean"]) / base_val) * 100
                baseline_row[f'{m}_imp'] = 0.0 # Pas de gain pour la baseline elle-même

        
        # Tri des modèles par performance RMSE moyenne
        others = others.sort_values(by="RMSE_mean", ascending=True)
        
        # Re-combinaison pour l'export (others puis baseline à la fin)
        final_df = pd.concat([others, baseline_row], ignore_index=True)
        self._export_to_latex(final_df, folder_path, exp_id)
        

    def _export_to_latex(self, df, folder_path, exp_id):
        if df.empty: return
        
        t_data = df['Target'].iloc[0]
        t_data.replace('subway_in', 'Subway-In').replace('subway_out', 'Subway-Out').replace('bike_in', 'Bike-In').replace('bike_out', 'Bike-Out').replace('_',' ')
        h = df['Horizon'].iloc[0]
        if len(self.contextual_dataset_names)>1:
            raise NotImplementedError("Multiple contextual datasets LaTeX export not implemented.")
        else:
            context_data = df['Context'].iloc[0].replace('subway_in', 'Subway-In').replace('subway_out', 'Subway-Out').replace('bike_in', 'Bike-In').replace('bike_out', 'Bike-Out').replace('_',' ')

        caption = (f"\\textbf{{Prediction {t_data.replace('_', ' ')} }} using contextual data {context_data.replace('_',' ')} according to different integration strategies at horizon (h{h}).")
        label = f"desag_{t_data}_h{h}"

        # --- MODIFICATION : Calcul des meilleurs (min) pour le gras ---
        best_values = {m: df[f"{m}_mean"].min() for m in self.metrics}

        col_spec = 'll' + 'c' * (2 * len(self.metrics))
        lines = [
            r"\begin{table}[H]", r"    \centering", r"    \resizebox{\textwidth}{!}{",
            fr"    \begin{{tabular}}{{{col_spec}}}", r"        \toprule",
            "        Model & Id & " + " & ".join([f"\\multicolumn{{2}}{{c}}{{{m}}}" for m in self.metrics]) + r" \\",
            "        & & " + " & ".join([r"Mean (Err. Red. \%)", "Std"] * len(self.metrics)) + r" \\",
            r"        \midrule"
        ]

        for i, row in df.iterrows():
            if row['Id'] == "Baseline" and i > 0:
                lines.append(r"        \midrule")
            
            cells = [row['Model'], row['Id']]
            for m in self.metrics:
                mean_val = row[f"{m}_mean"]
                std_val = row[f"{m}_std"]
                imp_val = row.get(f"{m}_imp", 0.0)
                
                # Vérification si c'est la meilleure valeur de la colonne
                is_best = (mean_val <= best_values[m])
                
                # Formatage de la cellule Mean
                if row['Id'] == "Baseline":
                    mean_str = f"{mean_val:.3f}"
                else:
                    mean_str = f"{mean_val:.3f} ({imp_val:+.1f}\\%)"
                
                if is_best:
                    mean_cell = f"\\textbf{{{mean_str}}}"
                    std_cell = f"\\textbf{{{std_val:.3f}}}" # Gras si Mean est best
                else:
                    mean_cell = mean_str
                    std_cell = f"{std_val:.3f}"
                
                cells.extend([mean_cell, std_cell])
            
            lines.append(f"        {' & '.join(cells)} \\\\")

        lines += [r"        \bottomrule", r"    \end{tabular}","}", 
                fr"    \caption{{{caption}}}", fr"    \label{{tab:{label}}}", r"\end{table}"]

        # Sauvegarde
        out_dir = os.path.join(folder_path, "latex_tables")
        os.makedirs(out_dir, exist_ok=True)

        # print('latex table: ')
        # print("\n".join(lines))
        with open(os.path.join(out_dir, f"{exp_id}.tex"), "w") as f:
            f.write("\n".join(lines))