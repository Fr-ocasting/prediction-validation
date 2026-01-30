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
from experiences.common_parameter import convertion_exp_name
from jupyter_ipynb.NetMob_training_analysis.plotting import plot_boxplot_on_metric


def tackle_trial_for_distrib(folder_path,dic_exp_to_names,L_metrics,exp_i,trial_j,metrics,
                             n_bis_range #=range(1,6)
                             ):
    df_j_all = pd.DataFrame()
    metric_i = []
    for n_bis in n_bis_range:
        df_j_all, metric_i = load_csv(folder_path,dic_exp_to_names,exp_i,trial_j,n_bis,df_j_all,metric_i,metrics)
    print('len(metric_i): ',len(metric_i))
    
    if len(metric_i) > 0: 
        metric_i = pd.DataFrame(metric_i)
        metric_i.index = [f"{trial_j}_bis{n_bis}" for n_bis in n_bis_range]
        L_metrics.append(metric_i)

    return L_metrics

def plotting_boxplot_of_trials(trials,exp_i,metrics,folder_path,
                               target_data,model_name,dataset_names,save_path,n_bis_range):
    L_metrics = []
    print(f"\nProcessing Experiment: {exp_i}")
    print("-----------------------")
    print(f"Trials to process: {trials}")
    dic_exp_to_names = {exp_i: f'{target_data}_{model_name}'}
    for trial_j in trials:
        L_metrics = tackle_trial_for_distrib(folder_path,dic_exp_to_names,L_metrics,exp_i,trial_j,metrics,n_bis_range)

    df_metrics_all = pd.concat(L_metrics)

    horizons = list(set([c.split('_')[-1][1:] for c in df_metrics_all.columns]))
    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(f"{save_path}/boxplot"):
                os.mkdir(f"{save_path}/boxplot")
    for horizon in horizons:
        print('\n---------------')
        print(f"Horizon: {horizon}")
        if save_path is not None:
            if not os.path.exists(f"{save_path}/boxplot/h{horizon}"):
                os.mkdir(f"{save_path}/boxplot/h{horizon}")
        df_horizon = df_metrics_all[[c for c in df_metrics_all.columns if c.endswith(f"_h{horizon}")]].dropna()

        # ----- Determine experiment name based on target_data and dataset_names

            
        exp_tmp = convertion_exp_name(target_data,dataset_names)
        # ------------

 
        df_horizon = update_df_metrics(df_horizon,exp_tmp)

        df_horizon['id'] = df_horizon['id'].apply(lambda x: x.replace('late_fusion_','L ').replace('CrossAttnBackBone_','CABB_').replace('BackBone_','BB_').replace('s_proj_t_proj','S-proj T-proj').replace('early_fusion_','E ').replace('independant_embedding','Indep Emb').replace('shared_embedding','Shared Emb').replace('traffic_model_backbone','Traffic Model BackBone').replace('simple_embedding','Simple Emb'))

        for metric in metrics: 
            metric = metric.lower()
            plot_boxplot_on_metric(df_horizon, metric_i=metric, xaxis_label="Config", legend_group='legend_group', width=1200, height=800,save_path=f"{save_path}/boxplot/h{horizon}/{metric}.html",bool_show=False,)
