import argparse

parser = argparse.ArgumentParser(description='STGformer')
parser.add_argument('--input_embedding_dim', type=str, default=24, help='Size of the embedding space for input features')
parser.add_argument('--tod_embedding_dim', type=str, default=12, help='Size of the Time-Of-Day embedding; set to 0 to disable')
parser.add_argument('--dow_embedding_dim', type=str, default=12, help='Size of the Day-Of‐Week embedding; set to 0 to disable')
parser.add_argument('--spatial_embedding_dim', type=str, default=0, help='Size of the fixed spatial embedding; set to 0 to disable')
parser.add_argument('--adaptive_embedding_dim', type=str, default=12, help='Size of the adaptive spatial embedding; set to 0 to disable')
parser.add_argument('--num_heads', type=str, default=4, help='Number of multi-head attention heads')
parser.add_argument('--num_layers', type=str, default=3, help='Number of MLP layers after attention')
parser.add_argument('--mlp_ratio', type=str, default=2, help='MLP hidden size ratio relative to model_dim')
parser.add_argument('--use_mixed_proj', type=str, default=True, help='Enable mixed projection (convolutions + linear) if True')
parser.add_argument('--dropout_a', type=str, default=0.3, help='Dropout rate applied to adaptive embeddings')
parser.add_argument('--adj_normalize_method', type=str, default='normlap',
                     choices = ['normlap','scalap','symadj','transition','doubletransition','identity'], help = 'Adjacency matrix normalization method.')
parser.add_argument('--kernel_size', type=str, default=[1], choices= [[1],[3],[1,1]], help='List of kernel sizes for projection in attention module.\
     There are as much SelfAttenLayer than kernel sizes.')

args = parser.parse_args(args=[])



parser_HP = argparse.ArgumentParser(description='HP')
parser_HP.add_argument('--HP_max_epochs', type=int, default=50, help="Number maximum of epochs per trial with ASHA Scheduler on Ray Tune")
parser_HP.add_argument('--weight_decay', type=float, default=0.05, help="weight decay for AdamW")
parser_HP.add_argument('--batch_size', type=int, default=32, help="Batch size")
parser_HP.add_argument('--lr', type=float, default=2e-3, help="Lr")
parser_HP.add_argument('--dropout', type=float, default=0.15, help="Dropout")
parser_HP.add_argument('--epochs', type=int, default=100, help="Epochs")
parser_HP.add_argument('--scheduler', type=bool, default=None, choices = [True, None], help="If True then acitvate a Lr scheduler with a warmup before reducing following an exponential function")
args_HP = parser_HP.parse_args(args=[])
# Other possible parameters: 
if False: 
adj_type,adj_normalize_method,threshold
    parser_HP.add_argument("--adj_type", type=str, default='corr', choices=['adj','dist','corr'], help="Adjacency matrix type")
    parser_HP.add_argument('--adj_normalize_method', type=str, default='normlap', choices = ['normlap','scalap','symadj','transition','doubletransition','identity'], help = 'Adjacency matrix normalization method.')
    parser_HP.add_argument('--threshold', type=float, default=0.7, help="Threshold for adjacency matrix and get a sparse one (non complete)")


