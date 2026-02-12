import argparse

parser = argparse.ArgumentParser(description='MTGNN')


parser.add_argument('--static_feat', default=None, 
                    help="Si = None, alors nodevec1 et nodevec2 sont  issues d'embedding different dans le graph constructor. Sinon Issue de static_feat qui est une matrice (statique pre-définie ?)")
parser.add_argument('--tanhalpha',type = int, default=3, 
                    help='Multiplicator coefficient used it GraphConstruction')
parser.add_argument('--node_dim', type = int, default=20,   #30
                    help="Embedding dim of each node. node_dim <= num_nodes (Cause we want to reduce dimensionality ?)")
parser.add_argument('--subgraph_size', type=int, default=15, #5 
                    help="Dimension du sous-graph. A priori <= node_dim car issue de matrice générée depuis l'embedding des noeuds")
parser.add_argument('--predefined_A', default=None, 
                    help='Set an precomputed Weighted Adjacency Matrix if needed (but then there is no graph construction)')
parser.add_argument('--propalpha', type=float, default=0.05, 
                    help='Parameter of the graph convolution')
parser.add_argument('--gcn_depth', type=int, default=2, 
                    help='Parameter of the graph convolution')

parser.add_argument('--buildA_true', type=bool, default=True, 
                    help=' Learn graph structure if True')
parser.add_argument('--gcn_true', type=bool, default=True, 
                    help=' Learn graph structure if True')
parser.add_argument('--layer_norm_affline', type=bool, default=True, 
                    help='????')
parser.add_argument('--layers', type=int, default=3, 
                    help='????')
parser.add_argument('--end_channels', type=int, default=64,  #128
                    help='????')
parser.add_argument('--skip_channels', type=int, default=64, 
                    help='????')
parser.add_argument('--residual_channels', type=int, default=32, 
                    help='????')
parser.add_argument('--conv_channels', type=int, default=32, 
                    help='????')
parser.add_argument('--c_in', type=int, default=1, 
                    help='In-Dimension of the first layer (Point-Wise Convolution)')
parser.add_argument('--dilation_exponential', type=int, default=1, 
                    help='As an impact on Kernel-Size of Skip convolutions')

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