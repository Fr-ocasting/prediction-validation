import argparse

parser = argparse.ArgumentParser(description='GraphWaveNet')


parser = argparse.ArgumentParser()

parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj') ???
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj') ???
parser.add_argument('--adj_type', default='adj', help="Type of adjacency matrix")  # Soit c'est adj de base, soit on fait random ou adj faut implémenter un truc 


parser.add_argument('--nhid',type=int,default=32,help='') ???
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension') ???

# parser.add_argument('--seed',type=int,default=99,help='random seed')

parser.add_argument('--filter_type', default='doubletransition', choices = ['laplacian', 'random_walk', 'dual_random_walk'], help="Filter type") AJOUTER DOUBLETRANSITION
parser.add_argument('--threshold', type=float, default=0.1,  # between 0.0 and 1.
                    help="threshold to mask the Weighted Adjacency Matrix based on Gaussian Kernel Distance. < threshold become 0")
args = parser.parse_args(args=[])

parser_HP = argparse.ArgumentParser(description='HP')
# parser_HP.add_argument('--HP_max_epochs', type=int, default=50, help="Number maximum of epochs per trial with ASHA Scheduler on Ray Tune")
parser_HP.add_argument('--weight_decay', type=float, default=0.0001, help="weight decay")
parser_HP.add_argument('--batch_size', type=int, default=64, help="Batch size")
parser_HP.add_argument('--lr', type=float, default=0.001, help="Lr")
parser_HP.add_argument('--dropout', type=float, default=0.3, help="Dropout")
parser_HP.add_argument('--epochs', type=int, default=100, help="Epochs")
# parser_HP.add_argument('--scheduler', type=bool, default=None, choices = [True, None], help="If True then acitvate a Lr scheduler with a warmup before reducing following an exponential function")
args_HP = parser_HP.parse_args(args=[])
# Other possible parameters: 


