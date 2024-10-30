import argparse

parser = argparse.ArgumentParser(description='MTGNN')


parser.add_argument('--static_feat', default=None, 
                    help="Si = None, alors nodevec1 et nodevec2 sont  issues d'embedding different dans le graph constructor. Sinon Issue de static_feat qui est une matrice (statique pre-définie ?)")

parser.add_argument('--tanhalpha',type = int, default=3, 
                    help='Set an precomputed Weighted Adjacency Matrix if needed (but then there is no graph construction)')

parser.add_argument('--node_dim', type = int, default=30, 
                    help="Dimension d'embedding. A priori <= num_nodes qui est définie dans utilities_DL.get_MultiModel_loss_args_emb_opts qui est la dimension d'embedding des noeuds")
parser.add_argument('--subgraph_size', type=int, default=5, 
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
parser.add_argument('--end_channels', type=int, default=128, 
                    help='????')
parser.add_argument('--skip_channels', type=int, default=64, 
                    help='????')
parser.add_argument('--residual_channels', type=int, default=32, 
                    help='????')
parser.add_argument('--conv_channels', type=int, default=32, 
                    help='????')
parser.add_argument('--c_in', type=int, default=1, 
                    help='????')
parser.add_argument('--dilation_exponential', type=int, default=1, 
                    help='????')

args = parser.parse_args(args=[])
