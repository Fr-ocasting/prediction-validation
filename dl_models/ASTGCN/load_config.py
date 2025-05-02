import argparse

parser = argparse.ArgumentParser(description='ASTGCN')

parser.add_argument('--nb_block', type=int, default=2, # 2 
                    help='Number of blocks in the model')
parser.add_argument('--K', type=int, default=2,   #3
                    help='Order of the Chebyshev polynomial')
parser.add_argument('--nb_chev_filter', type=int, default=32, #64 
                    help='Number of Chebyshev filters')
parser.add_argument('--nb_time_filter', type=int, default=64, # 64
                    help='Number of time filters')
parser.add_argument('--adj_type', type=str, default='corr', choices=['adj', 'dist','corr'], 
                    help='type of adjacency matrix')
parser.add_argument('--threshold', type=float, default=0.3, 
                    help='threshold to build sparse weighted adjacency matrix. Replace each value below threshold with 0. Is not used if adj_type == "adj"')
parser.add_argument('--time_strides', type=int, default=1, 
                    help='stride of the conv2D on the temporal dim')

args = parser.parse_args(args=[])

''' Config that comes from the original 'args_init' : '''
def transfer_from_orig_args(args_init,args):
    args.num_of_vertices = args_init.n_vertex
    args.num_for_predict = args_init.step_ahead
    args.len_input = args_init.L
    args.in_channels = args_init.C
    args.DEVICE = args_init.device
    return args




parser_HP = argparse.ArgumentParser(description='HP')
parser_HP.add_argument('--weight_decay', type=float, default=0.0005, help="weight decay for AdamW")
parser_HP.add_argument('--batch_size', type=int, default=32, help="Batch size")
parser_HP.add_argument('--lr', type=float, default=1e-3, help="Lr")
parser_HP.add_argument('--dropout', type=float, default=0.2, help="Dropout")
parser_HP.add_argument('--epochs', type=int, default=100, help="Epochs")
parser_HP.add_argument('--scheduler', type=bool, default=None, choices = [True, None], help="If True then acitvate a Lr scheduler with a warmup before reducing following an exponential function")
args_HP = parser_HP.parse_args(args=[])

# Init config to remove in this framework: 
if False:
    parser.add_argument('--adj_filename', type=str, default='./data/PEMS04/distance.csv', 
                        help='Path to the adjacency matrix file')
    parser.add_argument('--graph_signal_matrix_filename', type=str, default='./data/PEMS04/PEMS04.npz', 
                        help='Path to the graph signal matrix file')
    parser.add_argument('--ctx', type=int, default=0, 
                    help='Context or device ID (e.g., GPU ID)')
    parser.add_argument('--start_epoch', type=int, default=0, 
                    help='Starting epoch for training')
    parser.add_argument('--metric_method', type=str, default='unmask', 
                        help='Metric method for evaluation')
    parser.add_argument('--missing_value', type=float, default=0.0, 
                        help='Value to use for missing data')
    parser.add_argument('--model_name', type=str, default='astgcn_r', 
                        help='Name of the model')
    parser.add_argument('--points_per_hour', type=int, default=12, 
                        help='Number of points per hour in the dataset')
    parser.add_argument('--num_of_weeks', type=int, default=0, 
                        help='Number of weeks of data to use')
    parser.add_argument('--num_of_days', type=int, default=0, 
                        help='Number of days of data to use')
    parser.add_argument('--num_of_hours', type=int, default=1, 
                        help='Number of hours of data to use')


# Other possible parameters: 
if False: 
    parser_HP.add_argument("--momentum", type=float, default=0.95, help="momentum for SGD")
    parser_HP.add_argument('--scheduler', type=bool, default=True, help="If True then acitvate a Lr scheduler with a warmup before reducing following an exponential function")
    parser_HP.add_argument('--torch_scheduler_milestone', type=int, default=5, help="Number of epochs while we have a Lr warming up")
    parser_HP.add_argument('--torch_scheduler_gamma', type=float, default=0.99, help="Exponential coefficient associated to the lr decrease")
    parser_HP.add_argument('--torch_scheduler_lr_start_factor', type=float, default=0.2, help="Multiplicator coefficient of the lr for the first epoch, until reaching the value 'lr' at the epoch 'torch_scheduler_milestone")