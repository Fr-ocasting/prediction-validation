import numpy as np 
import torch
import torch.nn as nn
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True
import inspect
from typing import List, Dict, Optional, Tuple, Any
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
from dl_models.ASTGCN.ASTGCN import ASTGCN
from dl_models.ASTGCN.lib.utils import cheb_polynomial,scaled_Laplacian
from dl_models.STGformer.STGformer import STGformer

from utils.utilities import filter_args
from profiler.profiler import model_memory_cost
from build_inputs.load_adj import load_adj
import importlib

def load_vision_model(args_vision):
    script = importlib.import_module(f"dl_models.vision_models.{args_vision.model_name}.{args_vision.model_name}")
    importlib.reload(script)
    func = script.model
    filered_args = filter_args(func, args_vision)

    return func(**filered_args) 


def load_spatial_attn_model(args,query_dim,init_spatial_dim):
    script = importlib.import_module(f"dl_models.SpatialAttn.SpatialAttn")
    scrip_args = importlib.import_module(f"dl_models.SpatialAttn.load_config")
    importlib.reload(scrip_args)
    args_ds_i = scrip_args.args
    args_ds_i.dropout = args.dropout
    args_ds_i.query_dim = query_dim  # input dim of Query 
    args_ds_i.key_dim = init_spatial_dim  # input dim of Key 

    #print('\nquery/key dim : ',args_ds_i.query_dim,args_ds_i.key_dim)

    importlib.reload(script)
    func = script.model
    filered_args = filter_args(func, args_ds_i)

    return func(**filered_args)   

class full_model(nn.Module):
    
    ds_which_need_spatial_attn:  List[str]
    node_attr_which_need_attn:   List[str]
    dict_pos_node_attr2ds:       Dict[int, str]
    contextual_positions:        Dict[str, int]
    pos_calendar:             Dict[str,int]         


    def __init__(self,dataset, args):
        super(full_model,self).__init__()

        # Init for jit script 
        self.ds_which_need_spatial_attn = torch.jit.Attribute([], List[str])
        self.node_attr_which_need_attn  = torch.jit.Attribute([], List[str])
        self.dict_pos_node_attr2ds     = torch.jit.Attribute({}, Dict[int, str])
        self.dict_ds_which_need_attn2pos: Dict[str, List[int]] = torch.jit.Attribute({}, Dict[str, List[int]])
        self.pos_calendar = torch.jit.Attribute({}, Dict[str, int])

        # Init for the model
        self.ds_which_need_spatial_attn = list(args.ds_which_need_spatial_attn)
        self.node_attr_which_need_attn  = list(args.node_attr_which_need_attn)
        self.dict_pos_node_attr2ds     = dict(args.dict_pos_node_attr2ds)
        self.netmob_vision: Optional[nn.Module] = None
        self.te:            Optional[nn.Module] = None

        self.spatial_attn_by_station = nn.ModuleDict()   # même nom conservé
        self.spatial_attn_poi        = nn.ModuleDict()

        #___ Add positions for each dataset which need spatial attention:
        self.contextual_positions = args.contextual_positions

        #___ Add positions for each contextual data:
        self.pos_calendar: Dict[str,int] = {data_name: pos_i for data_name,pos_i in args.contextual_positions.items() if 'calendar' in data_name}

        #___ Add correspondance of position for each couple dataset/ list of positions:
        for pos,ds_name in self.dict_pos_node_attr2ds.items():     # ex: (pos,ds_name) = (2,'subway_out')
            if ds_name in self.node_attr_which_need_attn:        
                self.dict_ds_which_need_attn2pos[ds_name]= pos

        # Init Attention modules: 

        for ds_name in self.ds_which_need_spatial_attn:
            raise NotImplementedError('Dataset which need sptial attn have been implemented but issue with recent modifications')
            if ('netmob' in ds_name) or ('subway_out' in ds_name):
                init_dims = [getattr(args, f"n_units_{ds_name}_{k}") for k in range(len(self.contextual_positions[ds_name]))]
                self.spatial_attn_by_station[ds_name] = nn.ModuleList([ load_spatial_attn_model(args, query_dim=1, init_spatial_dim=d) for d in init_dims ])
            else:
                raise NotImplementedError(f"Dataset {ds_name} has not been implemented for spatial selection / spatial attention")

        for ds_name_i in self.node_attr_which_need_attn:
            init_dim = getattr(args, f"n_units_{ds_name_i}")
            self.spatial_attn_poi[ds_name_i] = load_spatial_attn_model(args, query_dim=args.n_vertex, init_spatial_dim=init_dim )

        # ...
        
        if dataset.target_data in args.dataset_names :
            self.remove_trafic_inputs = False
        else:
            self.remove_trafic_inputs = True
            print('\nPREDICTION WILL BE BASED SOLELY ON CONTEXTUAL DATA !\n')
        # ...

        # === Vision NetMob ===
        self.C = args.C
        args = self.update_vision_args(args)
        self.tackle_netmob(args)

        # === Tackle Node Graphe Attributes):
        self.tackle_node_attributes(args)

        # === Tackle module for spatial selection of contextual data: 
        self.build_spatial_attn_modules(args)

        # === TE ===
        self.te = TE_module(args).to(args.device) if 'calendar_embedding' in args.dataset_names else None
        if self.te is not None :
            self.TE_concatenation_early = args.args_embedding.concatenation_early 
            print('number of Parameters in Embedding Module: {}'.format(sum([p.numel() for p in self.te.parameters()])))
        else :
            self.TE_concatenation_early = False
            self.TE_concatenation_late = False
        if ((self.te is not None) and not(args.args_embedding.concatenation_early) and not(args.args_embedding.concatenation_late)):
                 raise ValueError('Calendar inputs but not taken into account. Need to set concatenation_early = True or concatenation_late = True')
        # === Trafic Model ===
        core_model, args = load_model(dataset, args)
        self.core_model = core_model

        self.n_vertex = args.n_vertex

    def update_vision_args(self,args):
        if len(vars(args.args_vision))>0:
            args.args_vision.n_vertex = args.n_vertex
            args.args_vision.H = args.H
            args.args_vision.W = args.W
            args.args_vision.dropout = args.dropout
            args.args_vision.x_input_size = args.L
        return args
    
    def tackle_netmob(self,args):
        # If 'netmob' is used as contextual data:
        if len(vars(args.args_vision))>0:
            self.netmob_vision = load_vision_model(args.args_vision)
            self.pos_netmob = args.contextual_positions[args.args_vision.dataset_name]
            self.vision_input_type = args.vision_input_type
            self.vision_concatenation_early = args.args_vision.concatenation_early
            print('number of Parameters in Vision Module: {}'.format(sum([p.numel() for p in self.netmob_vision.parameters()])))
            if (not(args.args_vision.concatenation_early) and not(args.args_vision.concatenation_late)):
                 raise ValueError('NetMob input but not taken into account. Need to set concatenation_early = True or concatenation_late = True')
        else:
            print('\nNetMob Vision is NONE')
            self.netmob_vision =  None
            self.vision_input_type = None
            self.vision_concatenation_early = False
            self.vision_concatenation_late = False
        
    def forward_calendar_model(self,x,contextual):
        if self.te is not None:
            time_elt = [contextual[pos]for _,pos in self.pos_calendar.items()]
            time_elt = [elt.long() for elt in time_elt] # time_elt.long()
            time_elt = self.te(x,time_elt) # Extract feature: [B] -> [B,C,N,L_calendar]

            if self.TE_concatenation_early:
                # Concat: [B,C,N,L],[B,C,N,L_calendar] -> [B,C,N,L+L_calendar]
                x = torch.cat([x,time_elt],dim = -1)


        elif (self.te is None) and (len(self.pos_calendar)>0):
            time_elt = torch.stack([contextual[pos]for _,pos in self.pos_calendar.items()],-1)
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
                if isinstance(self.pos_netmob, list):
                    extracted_feature = self.netmob_vision(x,[contextual[pos]for pos in self.pos_netmob])
                    # list of N tensor [B,Z] -> [B,N,Z]  And then unsqueeze :  [B,N,Z] -> [B,C,N,Z]
                    extracted_feature = torch.stack(extracted_feature,dim = 1).unsqueeze(1) 
                else:
                    extracted_feature = self.netmob_vision(x,contextual[self.pos_netmob])

            else:
                raise NotImplementedError(f"The Vision input type '{self.vision_input_type}' has not been implemented")
            
            # Concatenation early: 
            if self.vision_concatenation_early:
                # Concat: [B,C,N,L],[B,C,N,Z] -> [B,C,N,L+Z]
                if x.dim() == 4 and extracted_feature.dim()==3:
                    extracted_feature = extracted_feature.unsqueeze(1)
                x = torch.cat([x,extracted_feature],dim = -1)
        else:
            extracted_feature = None
        return x,extracted_feature
    
    def tackle_node_attributes(self,args):
        self.stacked_contextual = args.stacked_contextual
        self.n_vertex = args.n_vertex
        for dataset_name in self.ds_which_need_spatial_attn:
            position_i = getattr(args,f"pos_{dataset_name}")
            setattr(self,f"pos_{dataset_name}",position_i)
            for k,pos_i in enumerate(getattr(self,f"pos_{dataset_name}")):
                setattr(self,f"n_units_{dataset_name}_{k}",getattr(args,f"n_units_{dataset_name}_{k}"))

        for dataset_name_i in self.node_attr_which_need_attn: 
            setattr(self,f"n_units_{dataset_name_i}", getattr(args,f"n_units_{dataset_name_i}"))
        if ('netmob_POIs' in args.dataset_names) and (args.stacked_contextual) and (not args.compute_node_attr_with_attn):
            self.nb_add_channel = len(args.NetMob_selected_apps)*len(args.NetMob_transfer_mode)*len(args.NetMob_selected_tags) 

    def build_spatial_attn_modules(self,args):

        for ds_name in self.ds_which_need_spatial_attn:
            if ('netmob' in ds_name) or ('subway_out' in ds_name):
                init_spatial_dims = [getattr(args, f"n_units_{ds_name}_{k}") for k in range(len(getattr(args, f"pos_{ds_name}")))]
                self.spatial_attn_by_station[ds_name] = nn.ModuleList([
                    load_spatial_attn_model(args, query_dim=1, init_spatial_dim=init_spatial_dim)
                    for init_spatial_dim in init_spatial_dims
                ])
            else:
                raise NotImplementedError(f"Dataset {ds_name} has not been implemented for spatial selection / spatial attention")

        for ds in self.node_attr_which_need_attn:
            init_spatial_dim = getattr(args, f"n_units_{ds}")
            self.spatial_attn_poi[ds] = load_spatial_attn_model(args, query_dim=args.n_vertex, init_spatial_dim=init_spatial_dim)

            
    def stack_node_attribute(self,x: torch.Tensor, 
                             L_node_attributes: List[torch.Tensor]):
        ''' Concat node attributed to the channel dim of x'''

        #print('x.size: ',x.size())
        #print('node attributes size: ',[c.size() for c in L_node_attributes])
        x = torch.cat([x]+L_node_attributes, dim=1)
        return x
    

    def spatial_attention(self, x: torch.Tensor, 
                          contextual: List[torch.Tensor]):
        '''
        x:  [B,N,L]
        Some DataSet are not directly available as node attribute. 
        As example abbout POIs, we need spatial attention to reduce the channel dim to 1. 
        '''
        L_node_attributes: List[torch.Tensor] = torch.jit.annotate(List[torch.Tensor], [])
        #print('ds_which_need_spatial_attn: ',self.ds_which_need_spatial_attn)
        for ds_name, attn_list in self.spatial_attn_by_station.items():
            pos_list = self.contextual_positions[ds_name] 

            extracted_feature_for_spatial_unit_i: List[torch.Tensor] = []
            for k, pos_i in enumerate(pos_list):
                feat_k = attn_list[k](
                    x[:, k, :].unsqueeze(-1),                # [B,L]->[B,L,1]
                    contextual[pos_i].permute(0, 2, 1)         # [B,P,L]->[B,L,P]
                )                                            # => [B,L,Z]
                extracted_feature_for_spatial_unit_i.append(feat_k)

            # MultiHead-CrossAttention on Spatial Channel  between x[k] ([B,L]) and contextual_tensor[k] ([B,P,L]) -> Return [B,L,Z]
            # Spatial channel then need to unsqueeze and permute.
            extracted_feature_for_spatial_unit_i = torch.stack(extracted_feature_for_spatial_unit_i, dim=-1).permute(0,2,3,1)
            L_node_attributes.append(extracted_feature_for_spatial_unit_i)
        return L_node_attributes

    def add_other_node_attributes(self, 
                                  L_node_attributes: List[torch.Tensor], 
                                    x: torch.Tensor,
                                    contextual: List[torch.Tensor]
                                  ):
        '''
        Some 'Node attribute' doesnot need any spatial attention. 
        It is as example the case for 'subway-out' as contextual data for 'subway-in'. 
        We directly can attribute an attribute  
        '''
        for pos_i,ds_name_i in self.dict_pos_node_attr2ds.items():
            if not ds_name_i in self.node_attr_which_need_attn:
                node_attr = contextual[pos_i] 
                if ds_name_i == 'netmob_POIs':
                    #permute [B,P,L] ->  [B,L,P]  // reshape : [B,L,P] ->  [B,L,N,C]
                    node_attr = node_attr.permute(0, 2, 1)       # [B,L,P]
                    node_attr = node_attr.reshape(node_attr.size(0), node_attr.size(1), node_attr.size(2)//self.nb_add_channel,self.nb_add_channel).permute(0, 3, 2, 1) # [B,L,N,C]

                if node_attr.dim() == 3:
                    node_attr = node_attr.unsqueeze(1)

                L_node_attributes.append(node_attr)
            

        for ds_name_i,attetion_module_i in self.spatial_attn_poi.items():
            pos_i = self.dict_ds_which_need_attn2pos[ds_name_i] 
            node_attr = contextual[pos_i] 
            node_attr = attetion_module_i(x.permute(0, 2, 1),node_attr.permute(0, 2, 1))   # [B,L,Z*N]
            node_attr = node_attr.reshape(node_attr.size(0),node_attr.size(1),self.n_vertex, -1)  # [B,L,N,Z]
            node_attr = node_attr.permute(0, 3, 2, 1)    # [B,C,N,L]
            if node_attr is not None:
                if node_attr.dim() == 3:
                    node_attr = node_attr.unsqueeze(1)
                L_node_attributes.append(node_attr)
        return L_node_attributes

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
            #x = torch.Tensor().to(x)
            x = torch.empty(0, device=x.device, dtype=x.dtype)

        #print('x before stacking new channels:',x.size())
        # Spatial Attention and attributing node information: 
        if self.stacked_contextual: 
            L_node_attributes = self.spatial_attention(x,contextual)
            #print('L_node_attributes after spatial attn: ',[xb.size() for xb in L_node_attributes])
            L_node_attributes = self.add_other_node_attributes(L_node_attributes,x,contextual)
            #print('L_node_attributes after add_other_node_attributes: ',[xb.size() for xb in L_node_attributes])

            # [B,N,L] -> [B,1,N,L]
            if x.dim() == 3:
                x = x.unsqueeze(1)

            # [B,1,N,L] -> [B,C,N,L]
            x = self.stack_node_attribute(x,L_node_attributes)
        else:
            if x.dim() == 3:  # If only 1 channels
                x = x.unsqueeze(1)
        #print('x after attributing node information: ',x.size())
        x,extracted_feature = self.forward_netmob_model(x,contextual)        # Tackle NetMob (if exists):
        #print('x after NetMob model: ',x.size())
        #print('extracted_feature: ',extracted_feature.size() if extracted_feature is not None else None)
        x,time_elt = self.forward_calendar_model(x,contextual)         # Tackle Calendar Data (if exists)

        #print('x after Calendar model: ',x.size())
        #print('CalendarEmbedded Vector: ',time_elt.size() if time_elt is not None else None)

        # Core model 
        vision_late = self.vision_concatenation_late
        TE_late = self.TE_concatenation_late
        if self.core_model is not None:
            x= self.core_model(x,
                               extracted_feature if vision_late else None,
                               time_elt )
            #print('x after Core Model: ',x.size())
        # ...
        x = reshaping(x)
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
def reshaping(x):
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

def load_model(dataset, args):
    # Init L_add 
    if hasattr(args,'args_vision') and (len(vars(args.args_vision))>0):   #if not empty 
        # IF Early concatenation : 
        vision_concatenation_late = args.args_vision.concatenation_late
        vision_out_dim = args.args_vision.out_dim
        if args.args_vision.concatenation_early:
            if False:
                # Depend wether out_dim is implicit or defined by other parameters:
                if hasattr(args.args_vision,'out_dim'):
                    L_add = args.args_vision.out_dim
                else:
                    L_add = args.args_vision.L*args.args_vision.h_dim//2
            L_add = 7
            print('ATTENTION CHANGER LES LIGNES 380 DANS full_model.load_model()')
    else:
        L_add = 0
        vision_concatenation_late = False
        vision_out_dim = None

    if 'calendar_embedding' in args.dataset_names: #if not empty 
        # IF Early concatenation : 
        TE_concatenation_late = args.args_embedding.concatenation_late
        TE_embedding_dim = args.args_embedding.embedding_dim
        if args.args_embedding.concatenation_early:
            L_add = L_add + args.args_embedding.embedding_dim
    else:
        TE_concatenation_late = False
        TE_embedding_dim = None

    if args.model_name == 'STGformer':
        if TE_concatenation_late or vision_concatenation_late:
            raise NotImplementedError(f'{args.model_name} with TE_concatenation_late has not been implemented')
        from dl_models.STGformer.STGformer_utilities import normalize_adj_mx
        filtered_args = {k: v for k, v in vars(args).items() if (k in inspect.signature(STGformer.__init__).parameters.keys())}
        adj_mx,_ = load_adj(dataset,adj_type = args.adj_type, threshold= args.threshold)
        # normalze adjacency matrix : 
        adj_mx = normalize_adj_mx(adj_mx, args.adj_normalize_method, return_type="dense")

        supports = [torch.tensor(i).to(args.device) for i in adj_mx]
        model = STGformer(**filtered_args,supports=supports).to(args.device)
    if args.model_name == 'ASTGCN':
        if TE_concatenation_late or vision_concatenation_late:
            raise NotImplementedError(f'{args.model_name} with TE_concatenation_late has not been implemented')
        filtered_args = {k: v for k, v in vars(args).items() if k in inspect.signature(ASTGCN.__init__).parameters.keys()}
        adj_mx,_ = load_adj(dataset,adj_type = args.adj_type, threshold= args.threshold)
        L_tilde = scaled_Laplacian(adj_mx)
        cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(args.device) for i in cheb_polynomial(L_tilde, args.K)]
        #model = ASTGCN(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices)
        model = ASTGCN(**filtered_args,cheb_polynomials=cheb_polynomials).to(args.device)

    if args.model_name == 'TFT':
        if TE_concatenation_late or vision_concatenation_late:
            raise NotImplementedError(f'{args.model_name} with TE_concatenation_late has not been implemented')
        model = TFT(args)

    if args.model_name == 'MTGNN': 
        if TE_concatenation_late or vision_concatenation_late:
            raise NotImplementedError(f'{args.model_name} with TE_concatenation_late has not been implemented')
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
        if TE_concatenation_late or vision_concatenation_late:
            raise NotImplementedError(f'{args.model_name} with TE_concatenation_late has not been implemented')
        adj,_ = load_adj(dataset,adj_type = args.adj_type, threshold= args.threshold)
        model = DCRNN(adj, **vars(args)).to(args.device)
        
    if args.model_name == 'STGCN':
        from dl_models.STGCN.get_gso import get_output_kernel_size, get_block_dims, get_gso_from_adj
        Ko = get_output_kernel_size(args)
        args = get_block_dims(args,Ko)
        gso,_ = get_gso_from_adj(dataset, args)
        model = STGCN(args,gso=gso, blocks = args.blocks,Ko = Ko).to(args.device)
        
    if args.model_name == 'CNN': 
        model = CNN(args,L_add = L_add,vision_concatenation_late = False,TE_concatenation_late = False,vision_out_dim = None,TE_embedding_dim = None)

    if args.model_name in ['LSTM','GRU','RNN']:
        filtered_args = {k: v for k, v in vars(args).items() if k in inspect.signature(RNN.__init__).parameters.keys()}
        model = RNN(**filtered_args).to(args.device)

    if args.model_name == 'MLP':
        from dl_models.MLP.MLP import MLP_output 
        print('\n>>>>Model == MLP, keep in mind Concatenation Late DOES NOT WORK here. Only Concatenation Early')
        print(f'>>>>Also Stupid model. Input_dim = h_dim = L+L_add. Output_dim = {args.out_dim}')
        input_dim = args.L+L_add
        model = MLP_output(input_dim=input_dim,out_h_dim=input_dim,n_vertex=None,embedding_dim=args.out_dim,multi_embedding=False,dropout=args.dropout).to(args.device)

    if args.model_name == None:
        model = None

    if model is not None : model_memory_cost(model)
    return(model,args)