import argparse

parser = argparse.ArgumentParser(description='GMAN')


parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, default=1, help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8, help='number of attention heads')
parser.add_argument('--d', type=int, default=8, help='dims of each head attention outputs')

parser.add_argument('--decay_epoch', type=int, default=10,help='decay epoch')  ???
parser.add_argument('--SE_file', default='./data/SE(PeMS).txt', help='spatial embedding file')
parser.add_argument('--adj_type', default='adj', help="Type of adjacency matrix")
# parser.add_argument('--cl_decay_steps', type=int, default=1000, help="Curriculum learning decay steps. Not used if use_curriculum_learning=False")
# parser.add_argument('--use_curriculum_learning', type=bool, default=False, help="Learning method. Not needed here")
parser.add_argument('--filter_type', default='random_walk', choices = ['laplacian', 'random_walk', 'dual_random_walk'], help="Filter type") ???
parser.add_argument('--threshold', type=float, default=0.1,  # between 0.0 and 1.
                    help="threshold to mask the Weighted Adjacency Matrix based on Gaussian Kernel Distance. < threshold become 0")



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
# parser_HP.add_argument('--HP_max_epochs', type=int, default=50, help="Number maximum of epochs per trial with ASHA Scheduler on Ray Tune")
# parser_HP.add_argument('--weight_decay', type=float, default=0.0005, help="weight decay for AdamW")
parser_HP.add_argument('--batch_size', type=int, default=32, help="Batch size")
parser_HP.add_argument('--lr', type=float, default=0.001, help="Lr")
# parser_HP.add_argument('--dropout', type=float, default=0.2, help="Dropout")
# parser_HP.add_argument('--epochs', type=int, default=100, help="Epochs")
# parser_HP.add_argument('--scheduler', type=bool, default=None, choices = [True, None], help="If True then acitvate a Lr scheduler with a warmup before reducing following an exponential function")
args_HP = parser_HP.parse_args(args=[])
# Other possible parameters: 


