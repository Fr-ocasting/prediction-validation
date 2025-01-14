import numpy as np 
import torch
import torch.nn as nn
import inspect

# 
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

from utils.utilities import filter_args
from profiler.profiler import model_memory_cost
from build_inputs.load_adj import load_adj
from constants.paths import DATA_TO_PREDICT
import importlib

def load_vision_model(args_vision):
    script = importlib.import_module(f"dl_models.vision_models.{args_vision.model_name}.{args_vision.model_name}")
    importlib.reload(script)
    func = script.model
    filered_args = filter_args(func, args_vision)

    return func(**filered_args) 


class full_model(nn.Module):
    def __init__(self,dataset, args):
        super(full_model,self).__init__()

        # Add positions for each contextual data:
        if 'calendar' in args.contextual_positions.keys(): 
            self.pos_calendar = args.contextual_positions['calendar']

        if DATA_TO_PREDICT in args.dataset_names :
            self.remove_trafic_inputs = False
        else:
            self.remove_trafic_inputs = True
            print('\nPREDICTION WILL BE BASED SOLELY ON CONTEXTUAL DATA !\n')
        # ...

        
        # === Vision NetMob ===
        self.tackle_netmob(args)

        # === TE ===
        self.te = TE_module(args) if len(vars(args.args_embedding))>0 else None
        if self.te is not None :
            self.TE_concatenation_early = args.args_embedding.concatenation_early 
            print('number of Parameters in Embedding Module: {}'.format(sum([p.numel() for p in self.te.parameters()])))
        else :
            self.TE_concatenation_early = False
        if ((self.te is not None) and not(args.args_embedding.concatenation_early) and not(args.args_embedding.concatenation_late)):
                 raise ValueError('Calendar inputs but not taken into account. Need to set concatenation_early = True or concatenation_late = True')
        # === Trafic Model ===
        self.core_model = load_model(dataset, args)

        self.n_vertex = args.n_vertex

    def tackle_netmob(self,args):
        # If 'netmob' is used as contextual data:
        if len(vars(args.args_vision))>0:
            args.args_vision.n_vertex = args.n_vertex
            args.args_vision.H = args.H
            args.args_vision.W = args.W
            args.args_vision.dropout = args.dropout
            args.args_vision.x_input_size = args.L
            self.netmob_vision = load_vision_model(args.args_vision)
            self.vision_input_type = args.vision_input_type
            self.vision_concatenation_early = args.args_vision.concatenation_early
            self.pos_netmob = args.contextual_positions[args.args_vision.dataset_name]
            print('number of Parameters in Vision Module: {}'.format(sum([p.numel() for p in self.netmob_vision.parameters()])))
            if (not(args.args_vision.concatenation_early) and not(args.args_vision.concatenation_late)):
                 raise ValueError('NetMob input but not taken into account. Need to set concatenation_early = True or concatenation_late = True')
        else:
            self.netmob_vision =  None
            self.vision_input_type = None
            self.vision_concatenation_early = False
        
    def forward_te_model(self,x,contextual):
        if self.te is not None:
            time_elt = [contextual[pos]for pos in self.pos_calendar] # contextual[self.pos_calendar] 
            time_elt = [elt.long() for elt in time_elt]# time_elt.long()
            # Extract feature: [B] -> [B,C,N,L_calendar]
            time_elt = self.te(x,time_elt)
            #print('x',x.size())
            #print('time embedding + x passage dans gru',time_elt.size())
            if self.TE_concatenation_early:
                # Concat: [B,C,N,L],[B,C,N,L_calendar] -> [B,C,N,L+L_calendar]
                x = torch.cat([x,time_elt],dim = -1)
                #print('concat de x et  x + calendar',x.size())
        else:
            time_elt = None
            # ... 
        return x,time_elt  

    def forward_netmob_model(self,x,contextual):
        if self.netmob_vision is not None:
            if self.vision_input_type == 'image_per_stations':
                netmob_video_batch = contextual[self.pos_netmob]
                extracted_feature =  self.foward_image_per_stations(netmob_video_batch)
        
            elif self.vision_input_type == 'unique_image_through_lyon':
                netmob_video_batch = contextual[self.pos_netmob]
                extracted_feature =  self.forward_unique_image_through_lyon(netmob_video_batch)

            elif self.vision_input_type == 'POIs':
                # contextual[pos]: [B,nb_POIs,L']  // after forward : List of [B,Z] // After unsqueeze : [B,C,Z] with C = 1
                extracted_feature = self.netmob_vision(x,[contextual[pos]for pos in self.pos_netmob])
                # list of N tensor [B,Z] -> [B,N,Z]  And then unsqueeze :  [B,N,Z] -> [B,C,N,Z]
                extracted_feature = torch.stack(extracted_feature,dim = 1).unsqueeze(1)  


            else:
                raise NotImplementedError(f"The Vision input type '{self.vision_input_type}' has not been implemented")
            
            # Concatenation early: 
            if self.vision_concatenation_early:
                # Concat: [B,C,N,L],[B,C,N,Z] -> [B,C,N,L+Z]
                x = torch.cat([x,extracted_feature],dim = -1)
            
        else:
            extracted_feature = None

        return x,extracted_feature
    
    def reshaping(self,x):
        if x.dim()==4:
            x = x.squeeze()

        # Tackle the case when C=1 and output_dim = 1  
        if x.dim()==2:
            x = x.unsqueeze(-1)

        # Tackle the case when n_vertex = 1
        if x.dim()==1: 
            x = x.unsqueeze(-1)           
            x = x.unsqueeze(-1) 
        return x

    def forward(self,x,contextual = None):
        ''' 
        Args:
        -----
        x : 4-th order Tensor: Trafic Flow historical inputs [B,C,N,L]
        contextual : list of contextual data. 
            >>>> contextual[netmob_position]: [B,N,C,H,W,L]
            >>>> contextual[calendar]: [B]
        '''
        #print('\nx size before forward: ',x.size())
        if self.remove_trafic_inputs:
            x = torch.Tensor().to(x)
        else:
            if x.dim() == 3:
                x = x.unsqueeze(1)

        #print('x size after reshaping or reduction to 0: ',x.size())
        x,extracted_feature = self.forward_netmob_model(x,contextual)        # Tackle NetMob (if exists):
        #print('x after NetMob model: ',x.size())
        x,time_elt = self.forward_te_model(x,contextual)         # Tackle Calendar Data (if exists)
        #print('x after Calendar model: ',x.size())
        #print('CalendarEmbedded Vector: ',time_elt.size())

        # Core model 
        if self.core_model is not None:
            x= self.core_model(x,extracted_feature,time_elt)
            #print('x after Core Model: ',x.size())
        # ...
        x = self.reshaping(x)
        #print('x after reshaping: ',x.size())
        return(x)
    
    # =========================================================================== #
    ## ==== SHOULD BE USELESS FOR FUTURE ====================================
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
    ## ==========================================================================
    # =========================================================================== #


def load_model(dataset, args):

    # Init L_add 
    if hasattr(args,'args_vision') and (len(vars(args.args_vision))>0):   #if not empty 
        # IF Early concatenation : 
        vision_concatenation_late = args.args_vision.concatenation_late
        vision_out_dim = args.args_vision.out_dim
        if args.args_vision.concatenation_early:
            # Depend wether out_dim is implicit or defined by other parameters:
            if hasattr(args.args_vision,'out_dim'):
                L_add = args.args_vision.out_dim
            else:
                L_add = args.args_vision.L*args.args_vision.h_dim//2
    else:
        L_add = 0
        vision_concatenation_late = False
        vision_out_dim = None

    if hasattr(args,'args_embedding') and (len(vars(args.args_embedding))>0): #if not empty 
        # IF Early concatenation : 
        TE_concatenation_late = args.args_embedding.concatenation_late
        TE_embedding_dim = args.args_embedding.embedding_dim
        if args.args_embedding.concatenation_early:
            L_add = L_add + args.args_embedding.embedding_dim
    else:
        TE_concatenation_late = False
        TE_embedding_dim = None



    if args.model_name == 'TFT':
        model = TFT(args)

    if args.model_name == 'CNN': 
        model = CNN(args,L_add = L_add,vision_concatenation_late = False,TE_concatenation_late = False,vision_out_dim = None,TE_embedding_dim = None)

    if args.model_name == 'MTGNN': 
        filtered_args = {k: v for k, v in vars(args).items() if k in inspect.signature(MTGNN.__init__).parameters.keys()}
        model = MTGNN(**filtered_args,
                    L_add=L_add,
                    seq_length=args.L,
                    vision_concatenation_late = vision_concatenation_late,
                    TE_concatenation_late = TE_concatenation_late,
                    vision_out_dim = vision_out_dim,
                    TE_embedding_dim = TE_embedding_dim
                    ).to(args.device)
        
    if args.model_name == 'DCRNN':
        adj,_ = load_adj(dataset,adj_type = args.adj_type, threshold= args.threshold)
        model = DCRNN(adj, **vars(args)).to(args.device)
        
    if args.model_name == 'STGCN':
        from dl_models.STGCN.get_gso import get_output_kernel_size, get_block_dims, get_gso_from_adj
        Ko = get_output_kernel_size(args)
        blocks = get_block_dims(args,Ko)
        gso,_ = get_gso_from_adj(dataset, args)
        model = STGCN(args,gso=gso, blocks = blocks,Ko = Ko).to(args.device)

    if args.model_name == 'LSTM':
        from dl_models.LSTM.load_config import args as LSTM_args
        model = RNN(**vars(LSTM_args),out_dim =args.out_dim,L=args.L+L_add,dropout=args.dropout,device = args.device).to(args.device)
                          
    if args.model_name == 'GRU':
        from dl_models.GRU.load_config import args as GRU_args
        model = RNN(**vars(GRU_args),out_dim =args.out_dim,L=args.L+L_add, dropout=args.dropout,device = args.device).to(args.device)
   
    if args.model_name == 'RNN':
        from dl_models.RNN.load_config import args as RNN_args
        model = RNN(**vars(RNN_args),out_dim =args.out_dim,L=args.L+L_add,dropout=args.dropout,device = args.device).to(args.device)

    if args.model_name == 'MLP':
        from dl_models.MLP.MLP import MLP_output 
        print('\n>>>>Model == MLP, keep in mind Concatenation Late DOES NOT WORK here. Only Concatenation Early')
        print(f'>>>>Also Stupid model. Input_dim = h_dim = L+L_add. Output_dim = {args.out_dim}')
        input_dim = args.L+L_add
        model = MLP_output(input_dim=input_dim,out_h_dim=input_dim,n_vertex=None,embedding_dim=args.out_dim,multi_embedding=False,dropout=args.dropout).to(args.device)

    if args.model_name == None:
        model = None

    if model is not None : model_memory_cost(model)
    return(model)