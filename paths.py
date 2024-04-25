import os 

# Usual paths: 
folder_path = 'data/'

# Load subway in data:
#file_name = 'preprocessed_subway_15_min.csv'
#file_name = 'subway_IN_interpol_neg_15_min_16Mar2019_1Jun2020.csv'
file_name = 'subway_IN_interpol_neg_15_min_2019_2020.csv'

# Load CRITER data : 
#file_name = 'preprocessed_CRITER_6min.csv'

def get_save_directory(args): 

    # Common parameter between models:
    common_args_begin = f"save/{args.model_name}/K_fold{args.K_fold}/H{args.H}_D{args.D}_W{args.W}/"
    common_args_end  = f"opt{args.optimizer}/train_valid_calib_{args.train_prop}{args.valid_prop}{args.calib_prop}/E{args.epochs}_lr{args.lr}_B{args.batch_size}/"

    # Tackle different Models : 
    if args.model_name == 'STGCN':
        save_dir = f"{common_args_begin}{args.graph_conv_type}/{args.gso_type}/act_{args.act_fun}_Ks{args.Ks}/{common_args_end}"

    elif args.model_name == 'CNN':
        save_dir =  f"{common_args_begin}h_dims{'_'.join(list(map(str,args.H_dims)))}_out_dims{'_'.join(list(map(str,args.C_outs)))}/{common_args_end}"
    else:
        raise NotImplementedError(f'The model {args.model_name} has not been implemented in get_save_directory in paths.py')

    # Tackle case wether (or not) there is time embedding:
    if args.time_embedding:
        save_dir = f"{save_dir}FC1_17_8_FC2_8_4/Emb_dim{args.embedding_dim}/Specific_lr_{args.specific_lr}/CalendarClass{args.calendar_class}/position_{args.position}/"
    else:
        save_dir = f"{save_dir}no_embedding/"

    # Make directory is doesn't exist yet
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return(save_dir)

