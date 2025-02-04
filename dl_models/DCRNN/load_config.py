import argparse

parser = argparse.ArgumentParser(description='DCRNN')

parser.add_argument('--adj_type', default='dist', help="Type of adjacency matrix")
parser.add_argument('--cl_decay_steps', type=int, default=1000, help="Curriculum learning decay steps. Not used if use_curriculum_learning=False")
parser.add_argument('--use_curriculum_learning', type=bool, default=False, help="Learning method. Not needed here")
parser.add_argument('--input_dim', type=int, default=1, help="Number of input features (e.g., Flow, Velocity)")
parser.add_argument('--max_diffusion_step', type=int, default=3, choices = [1,2,3,4], help="Maximum diffusion steps. Might correspond to 'K' in the paper") 
parser.add_argument('--filter_type', default='random_walk', choices = ['laplacian', 'random_walk', 'dual_random_walk'], help="Filter type")
parser.add_argument('--num_rnn_layers', type=int, default=2, help="Number of RNN layers")
parser.add_argument('--rnn_units', type=int, default=32, help="Number of units per RNN layer")
parser.add_argument('--threshold', type=float, default=0.3,  # between 0.0 and 1.
                    help="threshold to mask the Weighted Adjacency Matrix based on Gaussian Kernel Distance. < threshold become 0")
args = parser.parse_args(args=[])

parser_HP = argparse.ArgumentParser(description='HP')
parser_HP.add_argument('--HP_max_epochs', type=int, default=50, help="Number maximum of epochs per trial with ASHA Scheduler on Ray Tune")
parser_HP.add_argument('--weight_decay', type=float, default=0.0005, help="weight decay for AdamW")
parser_HP.add_argument('--batch_size', type=int, default=32, help="Batch size")
parser_HP.add_argument('--lr', type=float, default=5e-2, help="Lr")
parser_HP.add_argument('--dropout', type=float, default=0.2, help="Dropout")
parser_HP.add_argument('--epochs', type=int, default=100, help="Epochs")
parser_HP.add_argument('--scheduler', type=bool, default=None, choices = [True, None], help="If True then acitvate a Lr scheduler with a warmup before reducing following an exponential function")
args_HP = parser_HP.parse_args(args=[])
# Other possible parameters: 
if False: 

    parser_HP.add_argument("--momentum", type=float, default=0.95, help="momentum for SGD")
    parser_HP.add_argument('--scheduler', type=bool, default=True, help="If True then acitvate a Lr scheduler with a warmup before reducing following an exponential function")
    parser_HP.add_argument('--torch_scheduler_milestone', type=int, default=5, help="Number of epochs while we have a Lr warming up")
    parser_HP.add_argument('--torch_scheduler_gamma', type=float, default=0.99, help="Exponential coefficient associated to the lr decrease")
    parser_HP.add_argument('--torch_scheduler_lr_start_factor', type=float, default=0.2, help="Multiplicator coefficient of the lr for the first epoch, until reaching the value 'lr' at the epoch 'torch_scheduler_milestone")