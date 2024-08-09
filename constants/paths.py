import os 
import torch 
# Usual paths: 
if torch.cuda.is_available():
    folder_path = '../../../../data/' 
else:
    folder_path = '../../../../Data/'


SAVE_DIRECTORY = '../save/'
# Load subway in data:
#file_name = 'preprocessed_subway_15_min.csv'
#file_name = 'subway_IN_interpol_neg_15_min_16Mar2019_1Jun2020.csv'
file_name = 'subway_IN_interpol_neg_15_min_2019_2020.csv'

# Load CRITER data : 
#file_name = 'preprocessed_CRITER_6min.csv'

results_folder = f"{SAVE_DIRECTORY}results/"
if not(os.path.exists(results_folder)):
    os.makedirs(results_folder)




# ================================================
# A SUPPRIMER 
# ================================================


def get_save_directory(args): 

    # Common parameter between models:
    common_args_begin = f"save/{args.loss_function_type}/{args.model_name}/K_fold{args.K_fold}/H{args.H}_D{args.D}_W{args.W}/"
    common_args_end  = f"E{args.epochs}_lr{args.lr}_B{args.batch_size}_train_valid_calib_{args.train_prop}{args.valid_prop}{args.calib_prop}/"

    # Tackle different Models : 
    if args.model_name == 'STGCN':
        save_dir = f"{args.graph_conv_type}_{args.gso_type}/act_{args.act_fun}_Ks{args.Ks}/"
    elif args.model_name == 'CNN':
        save_dir =  f"h_dims{'_'.join(list(map(str,args.H_dims)))}_out_dims{'_'.join(list(map(str,args.C_outs)))}/"
    elif args.model_name == 'MTGNN':
        save_dir = f"gcn_true{args.gcn_true}_conv{args.conv_channels}_res{args.residual_channels}_skip{args.skip_channels}_end{args.end_channels}_layer{args.layers}/gcn_depth{args.gcn_depth}_propalpha{args.propalpha}_subgraphsize{args.subgraph_size}_node_dim{args.node_dim}/"
    elif args.model_name =='DCRNN':
        save_dir = f"adj_type_{args.adj_type}_max_diffusion_step{args.max_diffusion_step}_filter_type_{args.filter_type}_num_nodes{args.num_nodes}_rnn_units{args.rnn_units}_num_rnn_layers{args.num_rnn_layers}/"

    else:
        raise NotImplementedError(f'The model {args.model_name} has not been implemented in get_save_directory in paths.py')
    
    save_dir = f"{common_args_begin}{save_dir}{common_args_end}"

    # Tackle case wether (or not) there is time embedding:
    if args.time_embedding:
        save_dir = f"{save_dir}TE_transfer_{args.TE_transfer}/Multi_Emb{args.multi_embedding}/FC1_17_8_FC2_8_4/Emb_dim{args.embedding_dim}/Specific_lr_{args.specific_lr}/CalendarClass{args.calendar_class}/position_{args.position}/"
    else:
        save_dir = f"{save_dir}no_embedding/"

    if args.epochs < 10:
        save_dir = 'petit_trial'
        

    # Make directory is doesn't exist yet
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return(save_dir)

