import argparse

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--Kt', type=int, default=3, choices=[2,3,4],
                    help='Kernel Size on the Temporal Dimension')

parser.add_argument('--stblock_num', type=int, default=2, choices=[2,3,4],
                    help='Number of STConv-blocks')

parser.add_argument('--Ks', type=int, default=2, choices=[1,2,3],
                    help='Number of iteration within the ChebGraphConv ONLY')

parser.add_argument('--graph_conv_type', type=str, default='graph_conv', 
                    choices = ['graph_conv','cheb_graph_conv'],
                    help='Type of graph convolution')

parser.add_argument('--gso_type', type=str, default='sym_norm_lap', 
                    choices = ['sym_norm_lap','rw_norm_lap','sym_renorm_adj','rw_renorm_adj'],
                    help='Type of calcul to compute the gso (Weighted Adjacency Matrix)')

parser.add_argument('--enable_bias', type=bool, default='True', 
                    choices = [True,False],
                    help='Enable bias on the output module (FC layers at the output of STGCN)')

parser.add_argument('--adj_type', type=str, default='corr', 
                    choices = ['adj','corr','dist'],
                    help='Type of Initial Adjacency Matrix (neighbors adjacency, distance correlation-based, or euclidian spatially-based distance)')

parser.add_argument('--enable_padding', type=bool, default=True,  
                    choices = [True,False],
                    help="Enable padding on the Temporal convolution. Suitable for short sequence cause (L' = L-2*(Kt-1)*stblock_num)")

parser.add_argument('--threshold', type=float, default=0.3,  # between 0.0 and 1.
                    help="threshold to mask the Weighted Adjacency Matrix based on Gaussian Kernel Distance. < threshold become 0")

parser.add_argument('--act_func', type=str, default='glu', 
                    choices = ['glu','gtu','silu'],
                    help="Type of activation function on the output module (FC layers at the output of STGCN)")

parser.add_argument('--temporal_h_dim', type=int, default=32, # 128 #64
                    choices = [8,16,32,64,128,256],
                    help="Dimension of temporal convolution. Stblocks dims = [temporal_h_dim, spatial_h_dim, temporal_h_dim]")

parser.add_argument('--spatial_h_dim', type=int, default=32, #32  #16 
                    choices = [8,16,32,64,128,256],
                    help="Dimension of spatial graph convolution. Stblocks dims = [temporal_h_dim, spatial_h_dim, temporal_h_dim]")

parser.add_argument('--output_h_dim', type=int, default=64, #128
                    choices = [8,16,32,64,128,256],
                    help="Dimension of hidden layers in output module")

parser.add_argument('--TGE_num_layers', type=int, default=2, 
                    choices = [1,2,3,4,8],
                    help="Number of Temporal Graph Encoder Layers if exist")

parser.add_argument('--TGE_num_heads', type=int, default=2, 
                    choices = [1,2,3,4,8],
                    help="Number of head in the Multi-Head Self-Attention module of each Temporal Graph Encoder Layers if exist. Have to devide 'temporal_h_dim'. dim_k = temporal_h_dim//n_heads")
parser.add_argument('--TGE_FC_hdim', type=int, default=32, 
                    choices = [1,2,3,4,8],
                    help="h-dim in the 2FC layer output in the TemporalGraphEncoder -->FC1(temporal_h_dim,TGE_FC_hdim) -- FC2FC1(TGE_FC_hdim,temporal_h_dim)")

                    

args = parser.parse_args(args=[])

# Def STGCN dim
def load_blocks(stblock_num,temporal_h_dim,spatial_h_dim,output_h_dim):
    blocks = []
    blocks.append([1])
    for l in range(stblock_num):
        blocks.append([temporal_h_dim, spatial_h_dim, temporal_h_dim])
    blocks.append([output_h_dim])
    return(blocks)
# ...
blocks = load_blocks(args.stblock_num,args.temporal_h_dim, args.spatial_h_dim,args.output_h_dim)
args.blocks = blocks

parser_HP = argparse.ArgumentParser(description='HP')
parser_HP.add_argument('--HP_max_epochs', type=int, default=50, help="Number maximum of epochs per trial with ASHA Scheduler on Ray Tune")
parser_HP.add_argument('--weight_decay', type=float, default=0.05, help="weight decay for AdamW")
parser_HP.add_argument('--batch_size', type=int, default=32, help="Batch size")
parser_HP.add_argument('--lr', type=float, default=2e-3, help="Lr")
parser_HP.add_argument('--dropout', type=float, default=0.15, help="Dropout")
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

