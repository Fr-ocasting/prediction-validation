import os 
import sys
import torch 
import pandas as pd

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..','..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from experiences.convert_df_to_latex import update_df_metrics,load_csv
from jupyter_ipynb.NetMob_training_analysis.plotting import plot_boxplot_on_metric


def tackle_trial_for_distrib(folder_path,dic_exp_to_names,L_metrics,exp_i,trial_j,metrics,
                             n_bis_range #=range(1,6)
                             ):
    df_j_all = pd.DataFrame()
    metric_i = []
    for n_bis in n_bis_range:
        df_j_all, metric_i = load_csv(folder_path,dic_exp_to_names,exp_i,trial_j,n_bis,df_j_all,metric_i,metrics)
    
    if len(metric_i) > 0: 
        metric_i = pd.DataFrame(metric_i)
        metric_i.index = [f"{trial_j}_bis{n_bis}" for n_bis in range(1,6)]
        L_metrics.append(metric_i)

    return L_metrics

def plotting_boxplot_of_trials(trials,exp_i,metrics,folder_path,dic_exp_to_names,save_path):
    for trial_j in trials:
        L_metrics = tackle_trial_for_distrib(folder_path,dic_exp_to_names,L_metrics,exp_i,trial_j,metrics)

    df_metrics_all = pd.concat(L_metrics)

    horizons = list(set([c.split('_')[-1][1:] for c in df_metrics_all.columns]))
    if save_path is not None:
        if not os.path.exists(f"{save_path}/boxplot"):
                os.mkdir(f"{save_path}/boxplot")
    for horizon in horizons:
        print('\n---------------')
        print(f"Horizon: {horizon}")
        if save_path is not None:
            if not os.path.exists(f"{save_path}/boxplot/h{horizon}"):
                os.mkdir(f"{save_path}/boxplot/h{horizon}")
        df_horizon = df_metrics_all[[c for c in df_metrics_all.columns if c.endswith(f"_h{horizon}")]].dropna()

        df_horizon = update_df_metrics(df_horizon,exp_i)

        for metric in metrics: 
            plot_boxplot_on_metric(df_horizon, metric_i=metric, xaxis_label="Config", legend_group='legend_group', width=1200, height=800,save_path=f"{save_path}/boxplot/h{horizon}/{metric}")
