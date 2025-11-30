import argparse

parser = argparse.ArgumentParser(description='GMAN')


parser = argparse.ArgumentParser()
parser.add_argument('--nb_STAttblocks', type=int, default=1, help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8, help='number of attention heads')
parser.add_argument('--d', type=int, default=8, help='dims of each head attention outputs')
parser.add_argument('--bn_decay', type=float, default=0.1, help='batch normalization decay rate')
parser.add_argument('--adj_type', default='dist', help="Type of adjacency matrix")


# parser.add_argument('--time_slot', type=int, default=5, help='a time step is 5 mins')
# parser.add_argument('--num_his', type=int, default=12, help='history steps')
# parser.add_argument('--num_pred', type=int, default=12, help='prediction steps')
# parser.add_argument('--train_ratio', type=float, default=0.7, help='training set [default : 0.7]')
# parser.add_argument('--val_ratio', type=float, default=0.1, help='validation set [default : 0.1]')
# parser.add_argument('--test_ratio', type=float, default=0.2, help='testing set [default : 0.2]')
# parser.add_argument('--max_epoch', type=int, default=1, help='epoch to run')
# parser.add_argument('--patience', type=int, default=10, help='patience for early stop')
# parser.add_argument('--traffic_file', default='./data/pems-bay.h5', help='traffic file')
# parser.add_argument('--model_file', default='./data/GMAN.pkl', help='save the model to disk')
# parser.add_argument('--log_file', default='./data/log', help='log file')

args = parser.parse_args(args=[])

parser_HP = argparse.ArgumentParser(description='HP')


parser_HP.add_argument('--optimizer', type=str, default='adam', choices = ['adam','Adam','sgd','AdamW'], help="Optimizer to use")
parser_HP.add_argument('--loss_function_type', type=str, default='MSE', help="Loss function to use")
parser_HP.add_argument('--batch_size', type=int, default=32, help="Batch size")
parser_HP.add_argument('--lr', type=float, default=0.001, help="Lr")
parser_HP.add_argument('--torch_scheduler_decay_epoch', type=int, default=10,help='decay epoch') 
parser_HP.add_argument('--torch_scheduler_gamma', type=float, default=0.9, help='StepLR gamma')
parser_HP.add_argument('--torch_scheduler_type', type=str, default='StepLR', help='Type of torch scheduler to use')

# parser_HP.add_argument('--dropout', type=float, default=0.2, help="Dropout")
# parser_HP.add_argument('--epochs', type=int, default=100, help="Epochs")
# parser_HP.add_argument('--scheduler', type=bool, default=None, choices = [True, None], help="If True then acitvate a Lr scheduler with a warmup before reducing following an exponential function")
args_HP = parser_HP.parse_args(args=[])
# Other possible parameters: 


