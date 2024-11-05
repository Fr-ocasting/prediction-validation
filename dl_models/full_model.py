import numpy as np 
import torch
import torch.nn as nn
import inspect

# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal import:
from dl_models.TimeEmbedding.time_embedding import TE_module
from dl_models.CNN.CNN_based_model import CNN
from dl_models.MTGNN.MTGNN import MTGNN
from dl_models.RNN.RNN import RNN
from dl_models.STGCN.STGCN import STGCN
from dl_models.DCRNN.DCRNN import DCRNN
from dl_models.TFT.TFT import TFT
from dl_models.STGCN.STGCN_utilities import calc_chebynet_gso,calc_gso

from dl_models.vision_models.simple_feature_extractor import FeatureExtractor_ResNetInspired,MinimalFeatureExtractor,ImageAvgPooling,FeatureExtractor_ResNetInspired_bis
from dl_models.vision_models.AttentionFeatureExtractor import AttentionFeatureExtractor
from dl_models.vision_models.FeatureExtractorEncoderDecoder import FeatureExtractorEncoderDecoder
from dl_models.vision_models.VideoFeatureExtractorWithSpatialTemporalAttention import VideoFeatureExtractorWithSpatialTemporalAttention

from profiler.profiler import model_memory_cost
from build_inputs.load_adj import load_adj
from argparse import Namespace
from constants.paths import DATA_TO_PREDICT

def filter_args(func, args):
    sig = inspect.signature(func)
    #valid_args = {k: v for k, v in args.items() if k in sig.parameters}
    filered_args = {k: v for k, v in vars(args).items() if k in sig.parameters}
    return filered_args


def load_vision_model(args_vision):
    if args_vision.model_name == 'ImageAvgPooling':
        filered_args = filter_args(ImageAvgPooling, args_vision)
        return ImageAvgPooling(**filered_args) 
    
    elif args_vision.model_name == 'MinimalFeatureExtractor':
        filered_args = filter_args(MinimalFeatureExtractor, args_vision)
        return MinimalFeatureExtractor(**filered_args)
    
    elif args_vision.model_name == 'FeatureExtractor_ResNetInspired':
        filered_args = filter_args(FeatureExtractor_ResNetInspired, args_vision)
        return FeatureExtractor_ResNetInspired(**filered_args)
                                   
    elif args_vision.model_name == 'FeatureExtractor_ResNetInspired_bis':
        filered_args = filter_args(FeatureExtractor_ResNetInspired_bis, args_vision)
        return FeatureExtractor_ResNetInspired_bis(**filered_args)

    elif args_vision.model_name == 'AttentionFeatureExtractor':
        filered_args = filter_args(AttentionFeatureExtractor, args_vision)
        return AttentionFeatureExtractor(**filered_args)

    elif args_vision.model_name == 'VideoFeatureExtractorWithSpatialTemporalAttention':
        filered_args = filter_args(VideoFeatureExtractorWithSpatialTemporalAttention, args_vision)
        return VideoFeatureExtractorWithSpatialTemporalAttention(**filered_args)

    elif args_vision.model_name == 'FeatureExtractorEncoderDecoder':
        filered_args = filter_args(FeatureExtractorEncoderDecoder, args_vision)
        return FeatureExtractorEncoderDecoder(**filered_args)
                                   
    else:
        NotImplementedError(f"Model {args_vision.model_name} has not been implemented")



class full_model(nn.Module):
    def __init__(self,args,dic_class2rpz):
        super(full_model,self).__init__()

        # === Vision NetMob ===
        if 'netmob' in args.contextual_positions.keys():
            args.args_vision.n_vertex = args.n_vertex
            args.args_vision.H = args.H
            args.args_vision.W = args.W
            self.netmob_vision = load_vision_model(args.args_vision)
            self.vision_input_type = args.vision_input_type
        else:
            self.netmob_vision =  None
            self.vision_input_type = None

        # === TE ===
        self.te = TE_module(args,args.args_embedding,dic_class2rpz) if args.time_embedding else None

        # === Trafic Model ===
        self.core_model = load_model(args,dic_class2rpz)

        self.n_vertex = args.n_vertex


        # Add positions for each contextual data:
        if 'calendar' in args.contextual_positions.keys(): 
            self.pos_calendar = args.contextual_positions['calendar']
        if 'netmob' in args.contextual_positions.keys(): 
            self.pos_netmob = args.contextual_positions['netmob']
        if DATA_TO_PREDICT in args.dataset_names :
            self.remove_trafic_inputs = False
        else:
            self.remove_trafic_inputs = True
            print('\nPREDICTION WILL BE BASED SOLELY ON CONTEXTUAL DATA !\n')
        # ...


    def foward_image_per_stations(self,netmob_video_batch):
        ''' Foward for input shape [B,C,N,H,W,L]'''
        B,N,C_netmob,H,W,L = netmob_video_batch.size()

        # Reshape:  [B,N,C,H,W,L] -> [B*N,C,H,W,L]
        netmob_video_batch = netmob_video_batch.reshape(B*N,C_netmob,H,W,L)

        # Forward : [B*N,C,H,W,L] ->  [B,C,N,Z]
        extracted_feature = self.netmob_vision(netmob_video_batch)

        # Reshape  [B*N,Z] -> [B,C,N,Z]
        extracted_feature = extracted_feature.reshape(B,self.n_vertex,-1)
        extracted_feature = extracted_feature.unsqueeze(1)

        return extracted_feature

    def forward_unique_image_through_lyon(self,netmob_video_batch):
        ''' Foward for input shape [B,C,H,W,L]'''
        B,C_netmob,H,W,L = netmob_video_batch.size()

        # Forward : [B,C,H,W,L] ->  [B,N*Z]
        extracted_feature = self.netmob_vision(netmob_video_batch)

        # [B,N*Z] ->  [B,N,Z]
        B,NZ = extracted_feature.size()
        Z = NZ//self.n_vertex
        extracted_feature = extracted_feature.view(B,self.n_vertex,Z) 
        # ...

        extracted_feature = extracted_feature.unsqueeze(1)   # [B,N,Z] ->  [B,1,N,Z]
        return extracted_feature
    

    def forward(self,x,contextual = None):
        ''' 
        Args:
        -----
        x : 4-th order Tensor: Trafic Flow historical inputs [B,C,N,L]
        contextual : list of contextual data. 
            >>>> contextual[netmob_position]: [B,N,C,H,W,L]
            >>>> contextual[calendar]: [B]
        '''
        if self.remove_trafic_inputs:
            x = torch.Tensor().to(x)
        else:
            if x.dim() == 3:
                x = x.unsqueeze(1)

        # if NetMob data is on :
        if self.netmob_vision is not None: 
            netmob_video_batch = contextual[self.pos_netmob]

            if self.vision_input_type == 'image_per_stations':
                extracted_feature =  self.foward_image_per_stations(netmob_video_batch)
        
            elif self.vision_input_type == 'unique_image_through_lyon':
                extracted_feature =  self.forward_unique_image_through_lyon(netmob_video_batch)
 
            else:
                raise NotImplementedError(f"The Vision input type '{self.vision_input_type}' has not been implemented")

            # Concat: [B,C,N,L],[B,C,N,Z] -> [B,C,N,L+Z]
            x = torch.cat([x,extracted_feature],dim = -1)
        # ...

        # if calendar data is on : 
        if self.te is not None:
            time_elt = contextual[self.pos_calendar].long()
            # Extract feature: [B] -> [B,C,N,L_calendar]
            time_elt = self.te(time_elt)
            
            # Concat: [B,C,N,L],[B,C,N,L_calendar] -> [B,C,N,L+L_calendar]
            x = torch.cat([x,time_elt],dim = -1)
        # ...


        # Core model 
        x = self.core_model(x)
        # ...

        return(x)


def load_model(args,dic_class2rpz):
    args_embedding =  args.args_embedding if hasattr(args,'args_embedding') else None
    if args.model_name == 'TFT':
        model = TFT(args)

    if args.model_name == 'CNN': 
        model = CNN(args,args_embedding = args_embedding,dic_class2rpz = dic_class2rpz)

    if args.model_name == 'MTGNN': 
        filtered_args = {k: v for k, v in vars(args).items() if k in inspect.signature(MTGNN.__init__).parameters.keys()}
        model = MTGNN(**filtered_args,
                    args_embedding=args_embedding,
                    seq_length=args.L,
                    ).to(args.device)
        
    if args.model_name == 'DCRNN':
        adj,_ = load_adj(adj_type = args.adj_type)
        model = DCRNN(adj, **vars(args)).to(args.device)
        
    if args.model_name == 'STGCN':
        from dl_models.STGCN.get_gso import get_output_kernel_size, get_block_dims, get_gso_from_adj
        Ko = get_output_kernel_size(args)
        blocks = get_block_dims(args,Ko)
        gso,_ = get_gso_from_adj(args)
        model = STGCN(args,gso=gso, blocks = blocks,Ko = Ko).to(args.device)

    if args.model_name == 'LSTM':
        from dl_models.LSTM.load_config import args as LSTM_args
        model = RNN(**vars(LSTM_args),L=args.L,dropout=args.dropout).to(args.device)
                          
    if args.model_name == 'GRU':
        from dl_models.GRU.load_config import args as GRU_args
        model = RNN(**vars(GRU_args),L=args.L, dropout=args.dropout).to(args.device)
   
    if args.model_name == 'RNN':
        from dl_models.RNN.load_config import args as RNN_args
        model = RNN(**vars(RNN_args),L=args.L,dropout=args.dropout).to(args.device)


    model_memory_cost(model)
    return(model)