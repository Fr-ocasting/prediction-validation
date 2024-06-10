import pandas as pd
from utilities_DL import get_DataSet_and_invalid_dates,get_MultiModel_loss_args_emb_opts
from DL_class import MultiModelTrainer
from config import get_args
from save_results import build_results_df
from paths import folder_path,file_name,get_save_directory


# ==== GET PARAMETERS ====
# Load config
model_name = 'STGCN' #'CNN' 
args = get_args(model_name)
#args = get_args(model_name = model_name,learn_graph_structure = True)  # MTGNN

# Modification :
# Args
args.epochs = 500
args.H = 6
args.W = 1
args.D = 1
args.K_fold = 6  # 66666666666666666666

args.TE_transfer = False
args.device = 'cuda:1'
args.lr = 1e-4
args.batch_size = 256
args.dropout = 0.2

# No Ray-Tuning, No Scheduler :
args.scheduler = None 
args.ray = False

def update_args(args):
    if args.loss_function_type == 'MSE':
        args.out_dim = 1
        args.alpha = None

    else:
        args.embedding_dim = 3
        args.calendar_class = 3
        args.position = 'input'
        args.specific_lr = False
        args.type_calendar = 'tuple'
        args.out_dim = 2
        args.alpha = 0.1

    return(args)




# Grid Search: 
multi_embeddings = [True,False,None,True,False,None] #None,
time_embeddings=  [True,True,False,True,True,False] #False,
loss_function_types = ['quantile','quantile','quantile','MSE','MSE','MSE'] 

for i,(multi_embedding,time_embedding,loss_function_type) in enumerate(zip(multi_embeddings,time_embeddings,loss_function_types)):
    
    args.multi_embedding = multi_embedding
    args.time_embedding = time_embedding
    args.loss_function_type = loss_function_type

    args = update_args(args)

    # Load dataset and invalid_dates 
    dataset,invalid_dates = get_DataSet_and_invalid_dates(args.abs_path,folder_path,file_name,args.W,args.D,args.H,args.step_ahead,single_station = args.single_station)
    (Datasets,DataLoader_list,time_slots_labels,dic_class2rpz,dic_rpz2class,nb_words_embedding) =  dataset.split_K_fold(args,invalid_dates)
    (loss_function,Model_list,Optimizer_list,Scheduler_list,args_embedding) = get_MultiModel_loss_args_emb_opts(args,nb_words_embedding,dic_class2rpz,n_vertex = len(Datasets[0].columns))


    # Keep only the 5 lasts datasets:
    Datasets,DataLoader_list,Model_list,Optimizer_list = Datasets[1:],DataLoader_list[1:],Model_list[1:],Optimizer_list[1:]
    
    # Load Multi-model trainer: 
    multimodeltrainer = MultiModelTrainer(Datasets,Model_list,DataLoader_list,args,Optimizer_list,loss_function, Scheduler_list,args_embedding,dic_class2rpz,show_figure= False) # Scheduler list = None
    (results_by_fold,mean_picp,mean_mpiw,dict_last_from_mean_of_folds,dict_best_from_mean_of_folds) = multimodeltrainer.K_fold_validation(mod_plot = 10)