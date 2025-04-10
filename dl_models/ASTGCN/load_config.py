import argparse

parser = argparse.ArgumentParser(description='ASTGCN')

parser.add_argument('--c_in', type=int, default=1, 
                    help='Channel dimension of the input. Usually = 1. As we consider trafic forecasting, we could set c_in = 2 if two time series are considered : Speed and Flow')
args = parser.parse_args(args=[])

parser_HP = argparse.ArgumentParser(description='HP')
parser_HP.add_argument('--weight_decay', type=float, default=0.0005, help="weight decay for AdamW")
parser_HP.add_argument('--batch_size', type=int, default=32, help="Batch size")
parser_HP.add_argument('--lr', type=float, default=5e-3, help="Lr")
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