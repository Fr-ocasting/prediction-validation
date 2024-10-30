import argparse
parser = argparse.ArgumentParser(description='LSTM')

parser.add_argument('--input_dim', type=int, default=1, choices = [1,2,3], help="Input Channel of LSTM. Example : 1 for Volume, 2 fo Volume and Speed ...")
parser.add_argument('--h_dim', type=int, nargs='+', default=16, choices = [8,16,32,64,128], help="Hidden layer dimensions")
parser.add_argument('--C_outs', type=int, nargs='+', default=[16,1], choices = [[16, 1], [32, 1], [16, 8, 1]], help="Output dimensions for convolution layers")
parser.add_argument('--num_layers', type=int, default=2, help="Number of layers")
parser.add_argument('--bias', type=bool, default=True, help="Use bias in layers")
parser.add_argument('--bidirectional', type=bool, nargs='+', default=[True, False], help="Enable bidirectional mode")
args = parser.parse_args(args=[])


