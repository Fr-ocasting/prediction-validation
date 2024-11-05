import argparse

parser = argparse.ArgumentParser(description='GRU')

parser.add_argument('--input_dim', type=int, default=1, choices = [1,2,3], help="Input Channel of LSTM. Example : 1 for Volume, 2 fo Volume and Speed ...")
parser.add_argument('--h_dim', type=int, nargs='+', default = 16, choices=[8,16, 32, 64,128], help="Hidden layer dimensions")
parser.add_argument('--C_outs', type=int, nargs='+', default=[16,1], choices=[[16,1],[32,1], [16, 8,1]], help="Output dimensions for convolution layers")
parser.add_argument('--num_layers', type=int, default=2, help="Number of layers")
parser.add_argument('--bias', type=bool, default=True, choices = [True, False], help="Use bias in layers")
parser.add_argument('--bidirectional', type=bool, nargs='+', default=True, choices = [True, False], help="Enable bidirectional mode")
parser.add_argument('--gru', type=bool, default=True, help="Enable lstm mode for rnn")
args = parser.parse_args(args=[])


parser_HP = argparse.ArgumentParser(description='HP')
parser_HP.add_argument('--HP_max_epochs', type=int, default=200, help="Number maximum of epochs per trial with ASHA Scheduler on Ray Tune")
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

