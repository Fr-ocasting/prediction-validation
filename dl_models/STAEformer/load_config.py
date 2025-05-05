import argparse

parser = argparse.ArgumentParser(description='STAEformer')
parser.add_argument('--input_embedding_dim', type=str, default=64, help='Size of the embedding space for input features')
parser.add_argument('--tod_embedding_dim', type=str, default=0, help='Size of the Time-Of-Day embedding; set to 0 to disable')
parser.add_argument('--dow_embedding_dim', type=str, default=0, help='Size of the Day-Of‐Week embedding; set to 0 to disable')
parser.add_argument('--spatial_embedding_dim', type=str, default=0, help='Size of the spatial embedding; is never used in the model')
parser.add_argument('--adaptive_embedding_dim', type=str, default=8, help='Size of the adaptive spatial embedding; set to 0 to disable')
parser.add_argument('--num_heads', type=str, default=1, help='Number of multi-head attention heads')
parser.add_argument('--num_layers', type=str, default=1, help='Number of MLP layers after attention')
parser.add_argument('--feed_forward_dim', type=str, default=256, help='Size of the feed forward layer after each attention layer (spatial and temporal) ')
parser.add_argument('--use_mixed_proj', type=bool, default=True, choices=[True, False], help='If True then use a mixed projection (FC on temporal dim  + FC on model dim) at the end of the model')
args = parser.parse_args(args=[])



parser_HP = argparse.ArgumentParser(description='HP')
parser_HP.add_argument('--weight_decay', type=float, default=0.05, help="weight decay for optimizer (L2 regularization if exists)")
parser_HP.add_argument('--batch_size', type=int, default=128, help="Batch size")
parser_HP.add_argument('--lr', type=float, default=2e-3, help="Lr")
parser_HP.add_argument('--dropout', type=float, default=0.15, help="Dropout for the spatial and temporal attention layers")
parser_HP.add_argument('--epochs', type=int, default=100, help="Epochs")
parser_HP.add_argument('--scheduler', type=bool, default=True, choices = [True, None], help="If True then acitvate a Lr scheduler with a warmup before reducing following an exponential function")
parser_HP.add_argument('--torch_scheduler_milestone', type=int, default=20)
parser_HP.add_argument('--torch_scheduler_gamma', type=float, default=0.1)
parser_HP.add_argument('--torch_scheduler_lr_start_factor', type=float, default=0.1)
args_HP = parser_HP.parse_args(args=[])
# Other possible parameters: 
if False: 
    parser_HP.add_argument("--adj_type", type=str, default='corr', choices=['adj','dist','corr'], help="Adjacency matrix type")
    parser_HP.add_argument('--adj_normalize_method', type=str, default='normlap', choices = ['normlap','scalap','symadj','transition','doubletransition','identity'], help = 'Adjacency matrix normalization method.')
    parser_HP.add_argument('--threshold', type=float, default=0.7, help="Threshold for adjacency matrix and get a sparse one (non complete)")


