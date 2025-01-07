import argparse

parser = argparse.ArgumentParser(description='VariableSelectionNetwork')

#parser.add_argument('--input_size', type=int, default=L, 
#                    help='Dimension of the input Temporal Dimension')

#parser.add_argument('--nb_channels', type=int, default=1+C,
#                    help='Number of contextual Time-Series (nb of POIs) + 1 Predicted Time Serie ')

parser.add_argument('--grn_h_dim', type=int, default=8, choices=[8,16,32,64,128,256],  # 16
                    help='Number of hidden dimension wihtin the GRU')

parser.add_argument('--grn_out_dim', type=int, default=32, choices=[8,16,32,64,128,256],  #  16
                    help='Output dimension of the enhanced vectors')

parser.add_argument('--contextual_static_dim', type=int, default=None, 
                    choices = [None,1,2,3],
                    help='Number of static variable information (such as calendar)')

parser.add_argument('--concatenation_late', type=bool, default=True, 
                    choices = [True,False],
                    help='If True then concatenation of extracted feature just before the output module of the backbone model')

parser.add_argument('--concatenation_early', type=bool, default=True, 
                    choices = [True,False],
                    help='If True then concatenation of extracted feature with the inputs at the begining of the backbone model.')  

parser.add_argument('--num_heads', type=bool, default=4, 
                    choices = [1,2,3,4,8],
                    help='number of head for the multi-head attention.') 

args = parser.parse_args(args=[])


def get_config(List_input_sizes,List_nb_channels):
    args.List_input_sizes = List_input_sizes
    args.List_nb_channels = List_nb_channels
    args.out_dim = args.grn_out_dim
    return(args)