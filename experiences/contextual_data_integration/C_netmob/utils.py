
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from pipeline.plotting.plotting import plot_coverage_matshow

# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..','..','..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

t_columns = ['t-D','t-6','t-5','t-4','t-3','t-2','t-1']


def plot_attention_score_STAEformer(trainer,ds,args,save, lmax = np.inf):
   # Forward on test set to get Attention weights: 
   all_data = [tensors for tensors in zip(*trainer.dataloader['test'])]
   X = torch.cat(all_data[0]).to(args.device)
   Y = torch.cat(all_data[1]).to(args.device)
   Xc = [torch.cat(context_tensors).to(args.device) for context_tensors in zip(*all_data[2])]
   model = trainer.model
   model.eval()
   with torch.no_grad():
      pred = model(X, Xc)
   # ---

   if hasattr(trainer.model,'_orig_mod'):
      t_layers = trainer.model._orig_mod.core_model.attn_layers_t
   else:
      t_layers = trainer.model.core_model.attn_layers_t
   for temporal_layer in range(min(lmax,len(t_layers))):
        attn_score = t_layers[temporal_layer].attn.attn_score.detach().cpu()     # attn_score.size : torch.Size([19956, 40, 7, 7]) - - [B,N,L,L] 
        print(attn_score.size(),'Std on dim 1: ',attn_score.std(1).mean().item(),'Min/Max: ',attn_score.mean(0).min().item(),attn_score.mean(0).max().item())
        mean_attn_score = attn_score.mean(0)                                                                             #  mean_attn_score.size(): [40, 7, 7]             - -   [N,L,L]
        df_attn_weight = pd.DataFrame(mean_attn_score.reshape(-1,mean_attn_score.shape[-1]).numpy(),columns=t_columns,index=[f"{hist} -- {station_name}"   for station_name in ds.spatial_unit for hist in t_columns] )
        figsize=(20,10)
        matfig = plt.figure(figsize=figsize)
        title = f"STAEformer - Temporal Attention Score - Layer {temporal_layer+1}"
        if save is not None: 
            save_i = save + f'STAEformer_T_attnscore_L{temporal_layer+1}.pdf'
        else:
            save_i = None
        plot_coverage_matshow(df_attn_weight, cmap ="YlOrRd", save = save_i, cbar_label =  "Number of Data",bool_reversed=False,
                                v_min=0,v_max=3*(1/df_attn_weight.shape[1]),
                                #   display_values = False,
                                    bool_plot = True,
                                    cbar_magic_args = True,
                                    figsize = figsize,
                                    title=title
        )


   if hasattr(trainer.model,'_orig_mod'):
      s_layers = trainer.model._orig_mod.core_model.attn_layers_s
   else:
      s_layers = trainer.model.core_model.attn_layers_s

   for temporal_layer in range(min(lmax,len(s_layers))):
        attn_score = s_layers[temporal_layer].attn.attn_score.detach().cpu()
        print(attn_score.size(),'Std on dim 1: ',attn_score.std(1).mean().item(),'Min/Max: ',attn_score.mean(0).min().item(),attn_score.mean(0).max().item())
        mean_attn_score = attn_score.mean(0)
        df_attn_weight = pd.DataFrame(mean_attn_score.reshape(-1,mean_attn_score.shape[-1]).numpy(),columns=ds.spatial_unit,index=[f"{hist} -- {station_name}"   for hist in t_columns  for station_name in ds.spatial_unit] )
        figsize=(20,10)
        matfig = plt.figure(figsize=figsize)
        title = f"STAEformer - Spatial Attention Score - Layer {temporal_layer+1}"
        if save is not None: 
            save_i = save + f'STAEformer_S_attnscore_L{temporal_layer+1}.pdf'
        else:
            save_i = None
        plot_coverage_matshow(df_attn_weight, cmap ="YlOrRd", save = save_i, cbar_label =  "Number of Data",bool_reversed=False,
                                    v_min=0,v_max=3*(1/df_attn_weight.shape[1]),
                                #   display_values = False,
                                    bool_plot = True,
                                    cbar_magic_args = True,
                                    figsize = figsize,title=title
        )




def plot_attention_score_CABB(trainer,ds,args,save,ds_name = 'netmob_POIs', lmax = np.inf):
   if hasattr(trainer.model.global_s_attn[f'{ds_name}'].model,'Q_attn_layers_t'):
      Q_attn_layers_t = trainer.model.global_s_attn[f'{ds_name}'].model.Q_attn_layers_t
      KV_attn_layers_t = trainer.model.global_s_attn[f'{ds_name}'].model.KV_attn_layers_t
      attn_layers_s = trainer.model.global_s_attn[f'{ds_name}'].model.attn_layers_s
   else:
      attn_layers_t = trainer.model.global_s_attn[f'{ds_name}'].model.attn_layers_t
      attn_layers_s = trainer.model.global_s_attn[f'{ds_name}'].model.attn_layers_s


   # --- Analysis of ST-Attention Weights associated to the contextual datasets: ---
   for mha_layer in range(min(lmax,len(Q_attn_layers_t))) :
      attn_score = Q_attn_layers_t[mha_layer].attn.attn_score.detach().cpu()
      mean_attn_score = attn_score.mean(0)
      df_attn_weight = pd.DataFrame(mean_attn_score.reshape(-1,mean_attn_score.size(-1)).numpy(),
                                 columns = t_columns,
                                 index = [f"{hist} -- {s_unit}" for hist in t_columns for s_unit in ds.spatial_unit] )  # args_init.contextual_kwargs['netmob_POIs']['spatial_unit']
      figsize=(20,10)
      matfig = plt.figure(figsize=figsize)
      title = f"Temporal - Q Attention Score - Layer {mha_layer+1}"
      if save is not None: 
         save_i = save + f'CABB_T_Q_attnscore_L{temporal_layer+1}.pdf'
      else:
         save_i = None
      plot_coverage_matshow(df_attn_weight, cmap ="YlOrRd", 
                           save = save_i, cbar_label =  "Number of Data",
                           bool_reversed=False, v_min=0,v_max=3*(1/df_attn_weight.shape[1]),
                           bool_plot = True, cbar_magic_args = True, 
                           figsize = figsize,title=title)
      
      # ---

   for mha_layer in range(min(lmax,len(KV_attn_layers_t))):
      attn_score = KV_attn_layers_t[mha_layer].attn.attn_score.detach().cpu()
      mean_attn_score = attn_score.mean(0)
      df_attn_weight = pd.DataFrame(mean_attn_score.reshape(-1,mean_attn_score.size(-1)).numpy(),
                                 columns = t_columns,
                                 index = [f"{hist} -- {s_unit}" for hist in t_columns for s_unit in args.contextual_kwargs['netmob_POIs']['spatial_unit']] )  # 
      title = f"Temporal - KV Attention Score - Layer {mha_layer+1}"
      if save is not None: 
         save_i = save + f'CABB_T_KV_attnscore_L{temporal_layer+1}.pdf'
      else:
         save_i = None
      plot_coverage_matshow(df_attn_weight, cmap ="YlOrRd", save = save_i, cbar_label =  "Number of Data",bool_reversed=False, v_min=0,v_max=3*(1/df_attn_weight.shape[1]),bool_plot = True, cbar_magic_args = True, figsize = figsize,title=title)
      # ---


   for mha_layer in range(min(lmax,len(attn_layers_s))): 
      attn_score = attn_layers_s[mha_layer].attn.attn_score.detach().cpu()
      mean_attn_score = attn_score.mean(0)
      df_attn_weight = pd.DataFrame(mean_attn_score.reshape(-1,mean_attn_score.size(-1)).numpy(),
                                 columns = args.contextual_kwargs[ds_name]['spatial_unit'],
                                 index =[f"{hist} -- {s_unit}"   for hist in t_columns for s_unit in ds.spatial_unit] )
      title = f"Spatial Attention Score - Layer {mha_layer+1}"
      if save is not None: 
         save_i = save + f'CABB_S_attnscore_L{temporal_layer+1}.pdf'
      else:
         save_i = None
      plot_coverage_matshow(df_attn_weight, cmap ="YlOrRd", save = save_i, cbar_label =  "Number of Data",bool_reversed=False, v_min=0,v_max=3*(1/df_attn_weight.shape[1]),bool_plot = True, cbar_magic_args = True, figsize = figsize, title=title)
      # ---