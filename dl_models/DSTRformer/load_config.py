import argparse

parser = argparse.ArgumentParser(description='DSTRformer')
parser.add_argument('--input_embedding_dim', type=int, default=8, help='Size of the embedding space for input features')
parser.add_argument('--tod_embedding_dim', type=int, default=4, help='Size of the Time-Of-Day embedding; set to 0 to disable')
parser.add_argument('--dow_embedding_dim', type=int, default=4, help='Size of the Day-Of‐Week embedding; set to 0 to disable')
parser.add_argument('--time_embedding_dim', type=int, default=0, help='Size of the Time-Series embedding; set to 0 to disable')
parser.add_argument('--ts_embedding_dim', type=int, default=8, help='Size of the Time-Series embedding; set to 0 to disable')
parser.add_argument('--adaptive_embedding_dim', type=int, default=8, help='Size of the adaptive spatial embedding; set to 0 to disable')
parser.add_argument('--node_dim', type=int, default=8, help='Size of the node embedding; set to 0 to disable')
parser.add_argument('--out_feed_forward_dim', type=int, default=64, help='Feed forward dim of the last attention layer before output proj')
parser.add_argument('--num_layers_m', type=int, default=1, help='Number of MLP layers after attention')
parser.add_argument('--mlp_num_layers', type=int, default=1, help='Number of MLP (2 layer) in fusion module of adj_mx information')
parser.add_argument('--num_heads', type=int, default=2, help='Number of heads in MHA on spatail attention and on temporal attention')
parser.add_argument('--num_layers', type=int, default=4, help='Number layers on spatail attention and on temporal attention ')
parser.add_argument('--feed_forward_dim', type=int, default=16, help='Size of the feed forward layer after each attention layer (spatial and temporal) ')
parser.add_argument('--use_mixed_proj', type=bool, default=True, choices=[True, False], help='If True then use a mixed projection (FC on temporal dim  + FC on model dim) at the end of the model')
parser.add_argument('--adj_type', type=str, default='adj', choices=['adj','dist','corr'], help="Adjacency matrix type")
parser.add_argument('--adj_normalize_method', type=str, default='doubletransition',
                    choices = ['normlap','scalap','symadj','transition','doubletransition','identity'], help = 'Adjacency matrix normalization method.')
parser.add_argument('--threshold', type=float, default=0.7, help="Threshold for adjacency matrix and get a sparse one (non complete)")
args = parser.parse_args(args=[])



parser_HP = argparse.ArgumentParser(description='HP')
parser_HP.add_argument('--weight_decay', type=float, default=0.066, help="weight decay for optimizer (L2 regularization if exists)")
parser_HP.add_argument('--batch_size', type=int, default=128, help="Batch size")
parser_HP.add_argument('--lr', type=float, default=1e-3, help="Lr")
parser_HP.add_argument('--dropout', type=float, default=0.15, help="Dropout for the spatial and temporal attention layers")
parser_HP.add_argument('--epochs', type=int, default=100, help="Epochs")
parser_HP.add_argument('--scheduler', type=bool, default=True, choices = [True, None], help="If True then acitvate a Lr scheduler with a warmup before reducing following an exponential function")
parser_HP.add_argument('--torch_scheduler_milestone', type=int, default=40)
parser_HP.add_argument('--torch_scheduler_gamma', type=float, default=0.995)
parser_HP.add_argument('--torch_scheduler_lr_start_factor', type=float, default=0.9)
args_HP = parser_HP.parse_args(args=[])
# Other possible parameters: 
if False: 
    parser_HP.add_argument("--adj_type", type=str, default='corr', choices=['adj','dist','corr'], help="Adjacency matrix type")
    parser_HP.add_argument('--adj_normalize_method', type=str, default='normlap', choices = ['normlap','scalap','symadj','transition','doubletransition','identity'], help = 'Adjacency matrix normalization method.')
    parser_HP.add_argument('--threshold', type=float, default=0.7, help="Threshold for adjacency matrix and get a sparse one (non complete)")


