import argparse

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--Kt', type=int, default=3, choices=[2,3,4],
                    help='Kernel Size on the Temporal Dimension')

parser.add_argument('--stblock_num', type=int, default=2, choices=[2,3,4],
                    help='Number of STConv-blocks')

parser.add_argument('--Ks', type=int, default=2, choices=[1,2,3],
                    help='Number of iteration within the ChebGraphConv ONLY')

parser.add_argument('--graph_conv_type', type=str, default='graph_conv', 
                    choices = ['graph_conv','cheb_graph_conv'],
                    help='Type of graph convolution')

parser.add_argument('--gso_type', type=str, default='sym_norm_lap', 
                    choices = ['sym_norm_lap','rw_norm_lap','sym_renorm_adj','rw_renorm_adj'],
                    help='Type of calcul to compute the gso (Weighted Adjacency Matrix)')

parser.add_argument('--enable_bias', type=bool, default='True', 
                    choices = [True,False],
                    help='Enable bias on the output module (FC layers at the output of STGCN)')

parser.add_argument('--adj_type', type=str, default='corr', 
                    choices = ['adj','corr','dist'],
                    help='Type of Initial Adjacency Matrix (neighbors adjacency, distance correlation-based, or euclidian spatially-based distance)')

parser.add_argument('--enable_padding', type=bool, default=True,  
                    choices = [True,False],
                    help="Enable padding on the Temporal convolution. Suitable for short sequence cause (L' = L-2*(Kt-1)*stblock_num)")

parser.add_argument('--threeshold', type=float, default=0.3,  # between 0.0 and 1.
                    help="Threeshold to mask the Weighted Adjacency Matrix. < threeshold become 0")

parser.add_argument('--act_func', type=str, default='glu', 
                    choices = ['glu','gtu','silu'],
                    help="Type of activation function on the output module (FC layers at the output of STGCN)")

args = parser.parse_args(args=[])

parser_HP = argparse.ArgumentParser(description='HP')
parser_HP.add_argument('--HP_max_epochs', type=int, default=50, help="Number maximum of epochs per trial with ASHA Scheduler on Ray Tune")
parser_HP.add_argument('--weight_decay', type=float, default=0.005, help="weight decay for AdamW")
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

