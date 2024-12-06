from plotting.TS_analysis import plot_single_point_prediction,plot_loss_from_trainer
from bokeh.layouts import row
from bokeh.plotting import show,output_notebook

# GET PARAMETERS
import sys
import os
import pandas as pd
import numpy as np 
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from examples.benchmark import local_get_args,get_trial_id,train_on_ds,keep_track_on_model_metrics
import matplotlib.pyplot as plt 
from K_fold_validation.K_fold_validation import KFoldSplitter

# Get df_True Volume: 
def get_df_for_visualisation(ds,Preds,Y_true,training_mode):
       df_verif = getattr(ds.tensor_limits_keeper,f"df_verif_{training_mode}")
       #df_true = pd.DataFrame(Y_true[:,list(ds.spatial_unit.index),0],columns = ds.spatial_unit.values,index = df_verif.iloc[:,-1])
       #df_prediction = pd.DataFrame(Preds[:,list(ds.spatial_unit.index),0],columns = ds.spatial_unit.values,index = df_verif.iloc[:,-1])
       df_true = pd.DataFrame(Y_true[...,0],columns = ds.spatial_unit,index = df_verif.iloc[:,-1])
       df_prediction = pd.DataFrame(Preds[...,0],columns = ds.spatial_unit,index = df_verif.iloc[:,-1])
       return(df_true,df_prediction)

def visualisation(trainer,ds,training_mode = 'test',station = ['CHA']):
    Preds,Y_true,T_labels = trainer.testing(ds.normalizer, training_mode =training_mode)
    df_true,df_prediction = get_df_for_visualisation(ds,Preds,Y_true,training_mode)
    col1 = plot_single_point_prediction(df_true,df_prediction,station=station,title = '',width=700,height=400,bool_show=False)
    col2 = plot_loss_from_trainer(trainer,width=500,height=400,bool_show=False)
    grid = row(col1,col2)

    output_notebook()
    show(grid)

def update_args_train_visu(model_name,name_ds,dataset_names,vision_model_name,dataset_for_coverage,modification,dic_class2rpz,df_loss,df_results,save_folder,station = ['CHA'],ds=None,args = None):
    print(f'\n>>>>Training {model_name} on {dataset_names}')
    if ds is None:
        ds = globals()[f"ds_{name_ds}"]
        args = globals()[f"args_{name_ds}"]  
    args_bis,folds,hp_tuning_on_first_fold = local_get_args(model_name,
                                                        args,
                                                        dataset_names=dataset_names,
                                                        dataset_for_coverage=dataset_for_coverage,
                                                        modification = modification)
    trial_id = get_trial_id(args_bis,vision_model_name=vision_model_name)
    trainer,df_loss = train_on_ds(model_name,ds,args_bis,trial_id,save_folder,dic_class2rpz,df_loss)
    metrics = trainer.metrics
    df_results = keep_track_on_model_metrics(trainer,df_results,model_name,trainer,metrics)
    visualisation(trainer,ds,training_mode = 'test',station = station)
    return trainer,df_results

def load_all(model_name,args,dataset_for_coverage,modification,vision_model_name,save_folder,station='CHA'):
    # Load DS: 
    df_loss,df_results = pd.DataFrame(),pd.DataFrame()

    # Load complete ds, no K-fold:
    K_fold_splitter = KFoldSplitter(args,vision_model_name,np.array([0]))
    ds,_,_,dic_class2rpz = K_fold_splitter.load_init_ds(normalize = True)

    # Analyses : 
    trainer,df_results = update_args_train_visu(model_name,None,
                                                args.dataset_names,vision_model_name,
                                                dataset_for_coverage,modification,
                                                dic_class2rpz,df_loss,
                                                df_results,save_folder,station,
                                                ds = ds, args= args)
    return ds,trainer,df_results