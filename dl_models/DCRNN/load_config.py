import argparse

parser = argparse.ArgumentParser(description='DCRNN')

parser.add_argument('--adj_type', default='dist', help="Type of adjacency matrix")
parser.add_argument('--cl_decay_steps', type=int, default=1000, help="Curriculum learning decay steps. Not used if use_curriculum_learning=False")
parser.add_argument('--use_curriculum_learning', type=bool, default=False, help="Learning method. Not needed here")
parser.add_argument('--input_dim', type=int, default=1, help="Number of input features (e.g., Flow, Velocity)")
parser.add_argument('--max_diffusion_step', type=int, default=2, choices = [1,2,3,4], help="Maximum diffusion steps")
parser.add_argument('--filter_type', default='random_walk', choices = ['laplacian', 'random_walk', 'dual_random_walk'], help="Filter type")
parser.add_argument('--num_rnn_layers', type=int, default=1, help="Number of RNN layers")
parser.add_argument('--rnn_units', type=int, default=1, help="Number of units per RNN layer")
args = parser.parse_args(args=[])

