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
results_df = pd.DataFrame()


# Modification :
# Args
args.epochs = 300
args.H = 6
args.W = 1
args.D = 1
args.K_fold = 5

args.embedding_dim = 3
args.calendar_class = 3
args.position = 'input'
args.specific_lr = False
args.type_calendar = 'tuple'

# Grid Search: 
transfers = [None,True,True,False,False]
multi_embeddings = [None,True,False,True,False]
time_embeddings=  [False,True,True,True,True]

for i,(transfer,multi_embedding,time_embedding) in enumerate(zip(transfers,
                                                              multi_embeddings,
                                                              time_embeddings)
                                                                    ):
  
  args.TE_transfer = transfer
  args.multi_embedding = multi_embedding
  args.time_embedding = time_embedding

  save_dir = get_save_directory(args)

  # Load dataset and invalid_dates 
  dataset,invalid_dates = get_DataSet_and_invalid_dates(folder_path,file_name,args.W,args.D,args.H,args.step_ahead,single_station = args.single_station)
  (Datasets,DataLoader_list,time_slots_labels,dic_class2rpz,dic_rpz2class,nb_words_embedding) =  dataset.split_K_fold(args,invalid_dates)

  # Load associated K_folds Models: 
  (loss_function,Model_list,Optimizer_list,args_embedding) = get_MultiModel_loss_args_emb_opts(args,nb_words_embedding,dic_class2rpz,n_vertex = len(Datasets[0].columns))
  multimodeltrainer = MultiModelTrainer(Datasets,Model_list,DataLoader_list,args,Optimizer_list,loss_function,scheduler = None,args_embedding=args_embedding,ray= False,save_dir = save_dir,dic_class2rpz=dic_class2rpz)

  (results_by_fold,mean_picp,mean_mpiw,dict_last_from_mean_of_folds,dict_best_from_mean_of_folds) = multimodeltrainer.K_fold_validation(mod_plot = 10)
  results_by_fold.to_csv(f"{save_dir}results_by_fold.csv")

  # Svae results 
  results_df = build_results_df(results_df,args, mean_picp,mean_mpiw,dict_last_from_mean_of_folds,dict_best_from_mean_of_folds)
  results_df.to_csv(f"{args.model_name}_H{args.H}_D{args.D}_W{args.W}_E{args.epochs}_K_fold{args.K_fold}_Emb_dim{args.embedding_dim}FC1_17_8_FC2_8_4_save_results.csv")