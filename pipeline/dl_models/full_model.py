import numpy as np 
import torch
from torch import Tensor
from typing import Optional,Tuple
import torch.nn as nn

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True
import inspect
from argparse import Namespace
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
from pipeline.dl_models.TimeEmbedding.time_embedding import TE_module
from pipeline.dl_models.CNN.CNN_based_model import CNN
from pipeline.dl_models.MTGNN.MTGNN import MTGNN
from pipeline.dl_models.RNN.RNN import RNN
from pipeline.dl_models.STGCN.STGCN import STGCN
from pipeline.dl_models.DCRNN.DCRNN import DCRNN
from pipeline.dl_models.TFT.TFT import TFT
from pipeline.dl_models.ASTGCN.ASTGCN import ASTGCN
from pipeline.dl_models.ASTGCN.lib.utils import cheb_polynomial,scaled_Laplacian
from pipeline.dl_models.STGformer.STGformer import STGformer
from pipeline.dl_models.STAEformer.STAEformer import STAEformer
from pipeline.dl_models.DSTRformer.DSTRformer import DSTRformer
from pipeline.dl_models.SARIMAX.SARIMAX import SARIMAX
from pipeline.dl_models.XGBoost.XGBoost import XGBoost
from pipeline.utils.utilities import filter_args
from pipeline.build_inputs.load_adj import load_adj
import importlib

# def load_feature_extractor_model(args_vision):
#     script = importlib.import_module(f"pipeline.dl_models.vision_models.{args_vision.model_name}.{args_vision.model_name}")
#     importlib.reload(script)  
#     func = script.model
#     filered_args = filter_args(func, args_vision)

#     return func(**filered_args) 


def load_spatial_attn_model(args,ds_name,query_dim,key_dim,output_temporal_dim = None,stack_consistent_datasets = False):
    # script = importlib.import_module(f"pipeline.dl_models.SpatialAttn.SpatialAttn")
    # scrip_args = importlib.import_module(f"pipeline.dl_models.SpatialAttn.load_config")
    # scrip_args = importlib.import_module(f"pipeline.dl_models.STAEformer.load_config")
    # importlib.reload(scrip_args)
    # args_ds_i = scrip_args.args
    # args_ds_i.dropout = args.dropout
    # args_ds_i.query_dim = query_dim  # input dim of Query 
    # args_ds_i.key_dim = key_dim  # input dim of Key 
    # args_ds_i.output_temporal_dim = output_temporal_dim 
    # args_ds_i.stack_consistent_datasets = stack_consistent_datasets
    
    script = importlib.import_module(f"pipeline.dl_models.STAEformer.STAEformer")
    args_ds_i = {}
    args_ds_i = Namespace(**args_ds_i)
    for key,value in args.contextual_kwargs[ds_name]['attn_kwargs'].items():
        setattr(args_ds_i,key,value)

    importlib.reload(script)
    # func = script.model
    func = script.MultiLayerCrossAttention
    filered_args = filter_args(func, args_ds_i)

    return func(**filered_args)   

class full_model(nn.Module):
    
    contextual_positions:        Dict[str, int]
    pos_calendar:                Dict[str,int]    
    ds_which_need_spatial_attn_per_station:  List[str]
    ds_which_need_global_attn:   List[str]
    dict_pos_node_attr2ds:       Dict[int, str]
    dict_ds_which_need_attn2pos: Dict[str, List[int]]
    # nb_add_channel:              int
    dic_stacked_contextual:      Dict[str,bool]


    """
    ds_which_need_spatial_attn_per_station: List of dataset names which need sub-attention module at each spatial unit.
        >>> ex:  Selection of Pi pois around a station i  and then attention to reduce the Pi time serie to a unique channel.


    ds_which_need_global_attn:  List of dataset names which need a spatial attention
        >>> ex : Selection of P spatiaal units and then attention to reduce the P time-series to a N*Cp tmie-serie. 
    """

    def __init__(self,dataset, args):
        super(full_model,self).__init__()

        # Init for jit script 
        # self.nb_add_channel = torch.jit.Attribute(0, int)

        # ------- Init for the model
        self.dict_pos_node_attr2ds                              = args.dict_pos_node_attr2ds
        self.dict_pos_node_attr2ds_keys                         = list(self.dict_pos_node_attr2ds.keys())

        # Global Attn:
        self.ds_which_need_global_attn                          = list(args.ds_which_need_global_attn)
        self.dict_ds_which_need_attn2pos                        = {}

        # Global Attn Late:
        self.ds_which_need_global_attn_late                     = list(args.ds_which_need_global_attn_late)
        self.dict_ds_which_need_attn_late2pos                   = {}

        # Local Attn:
        self.ds_which_need_spatial_attn_per_station             = list(args.ds_which_need_spatial_attn_per_station)

        # Will be stacked 
        self.dict_pos_node_attr_which_does_not_need_attn2ds     = {ds_name:p for p,ds_name in self.dict_pos_node_attr2ds.items() if ((p not in self.ds_which_need_global_attn) and 
                                                                                                                                     (p not in self.ds_which_need_global_attn_late) and
                                                                                                                                     ('stacked_contextual' in args.contextual_kwargs[ds_name].keys()) and
                                                                                                                                     (args.contextual_kwargs[ds_name]['stacked_contextual'])
                                                                                                                                     )}
        self.pos_node_attr_which_does_not_need_attn             = list(self.dict_pos_node_attr_which_does_not_need_attn2ds.values())
        self.dict_ds2added_dim                                  = {}

        self.netmob_vision= None 
        self.te= None        

        self.spatial_attn_per_station = nn.ModuleDict()   # même nom conservé
        self.global_s_attn        = nn.ModuleDict()
        self.dic_stacked_contextual = {}
        self.dic_stacked_consistant_contextual = {}
        # ------- 

        #___ Add positions for each dataset which need spatial attention:
        self.contextual_positions = args.contextual_positions

        #___ Add positions for each contextual data:
        self.pos_calendar = {data_name: pos_i for data_name,pos_i in args.contextual_positions.items() if 'calendar' in data_name}

        #___ Add correspondance of position for each couple dataset/ list of positions:
        for pos,ds_name in self.dict_pos_node_attr2ds.items():     # ex: (pos,ds_name) = (2,'subway_out')
            if ds_name in self.ds_which_need_global_attn:        
                self.dict_ds_which_need_attn2pos[ds_name]= pos
            if ds_name in self.ds_which_need_global_attn_late:
                self.dict_ds_which_need_attn_late2pos[ds_name]= pos
        args.dict_ds_which_need_attn_late2pos = self.dict_ds_which_need_attn_late2pos


            
        if dataset.target_data in args.dataset_names :
            self.remove_trafic_inputs = False
        else:
            self.remove_trafic_inputs = True
            print('\n------------------------------------------------------------------------\nPREDICTION WILL BE BASED SOLELY ON CONTEXTUAL DATA !\n')
        # ...

        """ # To remove : 
        # === Feature Extractor Module ===
        self.C = args.C
        args = self.update_vision_args(args)
        self.tackle_netmob(args)
        """
        # === Tackle Feature Extractor:
        for ds_name,kwargs in args.contextual_kwargs.items():
            self.dic_stacked_contextual[ds_name] = kwargs['stacked_contextual']
            self.dic_stacked_consistant_contextual[ds_name] = kwargs['stack_consistent_datasets'] if 'stack_consistent_datasets' in kwargs.keys() else False
            if 'added_dim' in kwargs.keys():
                self.dict_ds2added_dim[ds_name] = kwargs['added_dim']

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
        args.time_step_per_hour = int(dataset.time_step_per_hour)
        core_model, args = load_model(dataset, args)
        self.core_model = core_model
        self.L = args.L
        self.num_nodes = args.num_nodes

        
    def forward_calendar_model(self,x:Tensor,
                               contextual:List[Tensor])->Tuple[Tensor,Optional[Tensor]]:
        if self.te is not None:
            x_calendar = [contextual[pos]for _,pos in self.pos_calendar.items()]
            x_calendar = [elt.long() for elt in x_calendar] # x_calendar.long()
            x_calendar = self.te(x,x_calendar) # Extract feature: [B] -> [B,C,N,L_calendar]

            if self.TE_concatenation_early:
                # Concat: [B,C,N,L],[B,C,N,L_calendar] -> [B,C,N,L+L_calendar]
                x = torch.cat([x,x_calendar],dim = -1)
        elif (self.te is None) and (len(self.pos_calendar)>0):
            x_calendar = torch.stack([contextual[pos]for _,pos in self.pos_calendar.items()],-1)
        else:
            x_calendar = None
            # ... 

        return x,x_calendar  
 
    def tackle_node_attributes(self,args):
        for dataset_name in self.ds_which_need_spatial_attn_per_station:
            position_i = getattr(args,f"pos_{dataset_name}")
            setattr(self,f"pos_{dataset_name}",position_i)
            for k,pos_i in enumerate(getattr(self,f"pos_{dataset_name}")):
                setattr(self,f"n_units_{dataset_name}_{k}",getattr(args,f"n_units_{dataset_name}_{k}"))

        for dataset_name_i in self.ds_which_need_global_attn: 
            setattr(self,f"n_units_{dataset_name_i}", getattr(args,f"n_units_{dataset_name_i}"))

        # if ('netmob_POIs' in args.contextual_dataset_names) and \
        #     (args.contextual_kwargs['netmob_POIs']['stacked_contextual']) and \
        #     (not args.contextual_kwargs['netmob_POIs']['need_global_attn']):
        #     self.nb_add_channel = len(args.contextual_kwargs['netmob_POIs']['NetMob_selected_apps'])*len(args.contextual_kwargs['netmob_POIs']['NetMob_transfer_mode'])*len(args.contextual_kwargs['netmob_POIs']['NetMob_selected_tags']) 

    def build_spatial_attn_modules(self,args):
        for ds_name in self.ds_which_need_spatial_attn_per_station:
            if ('netmob' in ds_name) or ('subway_out' in ds_name):
                init_spatial_dims = [getattr(args, f"n_units_{ds_name}_{k}") for k in range(len(getattr(args, f"pos_{ds_name}")))] # range(len(self.contextual_positions[ds_name]))
                self.spatial_attn_per_station[ds_name] = nn.ModuleList([
                    load_spatial_attn_model(args,ds_name, query_dim=1, key_dim=init_spatial_dim)
                    for init_spatial_dim in init_spatial_dims
                ])
            else:
                raise NotImplementedError(f"Dataset {ds_name} has not been implemented for spatial selection / spatial attention")
            
        for ds_name in self.ds_which_need_global_attn:
            # if ('attn_kwargs' in args.contextual_kwargs[ds_name].keys()) and ('L_out' in args.contextual_kwargs[ds_name]['attn_kwargs'].keys()):
            #     L_out = args.contextual_kwargs[ds_name]['attn_kwargs']['L_out']
            # else:
            #     L_out = None
            # condition_i = ('attn_kwargs' in args.contextual_kwargs[ds_name].keys()) and ('stack_consistent_datasets' in args.contextual_kwargs[ds_name].keys())
            self.global_s_attn[ds_name] = load_spatial_attn_model(args,ds_name, 
                                                                     query_dim=1 if hasattr(args.contextual_kwargs[ds_name],'keep_temporal_dim') and args.contextual_kwargs[ds_name]['keep_temporal_dim'] else args.L, 
                                                                     key_dim=1 if hasattr(args.contextual_kwargs[ds_name],'keep_temporal_dim') and args.contextual_kwargs[ds_name]['keep_temporal_dim'] else args.L,
                                                                    #  output_temporal_dim = L_out,
                                                                    #  stack_consistent_datasets = args.contextual_kwargs[ds_name]['stack_consistent_datasets'] if condition_i else False
                                                                     )
        self.KEY_global_s_attn = list(self.global_s_attn.keys())

            
    def stack_node_attributes_from_attn(self,x: torch.Tensor, 
                             L_node_attributes: List[torch.Tensor],
                             L_projected_x:List[torch.Tensor], )-> Tensor:
    
        ''' Concat node attributed to the channel dim of x'''

        # print('x.size: ',x.size())
        # print('node attributes size: ',[c.size() for c in L_node_attributes])
        if len(L_projected_x) != 0:
            x = torch.cat(L_projected_x,dim = 1)
            
        if len(L_node_attributes) != 0:
            x = torch.cat([x]+L_node_attributes, dim=1)
        # print('output.size: ',x.size())
        return x
    

    def local_spatial_attention(self, x: torch.Tensor, 
                          contextual: List[torch.Tensor])-> List[torch.Tensor]:
        '''
        Compute Spatial Attention Localy between each spatial unit of the target data and all the spatial units of the contextual data.
        Spatial Attention has to be computed on Heterogenous spatial units, but can still be used for Homogenous spatial units

        Args: 
        -------
        x:  [B,N,L]
        contextual : List of contextual data tensors [B,Pi,L] (where Pi is the number of spatial units in the contextual data i)

        For each n in N : 
            extract features from contextual data : [B,Pi,L] -> [B,1,L]
        Stack them all: 
            [[B,1,L],...,[B,1,L] ] -> [B,1,N,L]

        Outputs: 
        ---------
        The contextual dataset has to be stacked as inputs: 
        or 
        The contextual dataset has to be concatenated before the output module
        '''

        #print('Spatial Attention for Node Attributes')
        L_node_attributes   : List[torch.Tensor] = torch.jit.annotate(List[torch.Tensor], [])
        L_extracted_feature : List[torch.Tensor] = torch.jit.annotate(List[torch.Tensor], [])

        #print('ds_which_need_spatial_attn_per_station: ',self.ds_which_need_spatial_attn_per_station)
        # print('self.spatial_attn_per_station.keys(): ',self.spatial_attn_per_station.keys())
        # print('self.dic_stacked_contextual:', self.dic_stacked_contextual)
        for ds_name, attn_list in self.spatial_attn_per_station.items():
            # print('Dataset: ',ds_name)
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

            if self.dic_stacked_contextual[ds_name]:
                L_node_attributes.append(extracted_feature_for_spatial_unit_i)
            else:
                L_extracted_feature.append(extracted_feature_for_spatial_unit_i)
        if len(L_extracted_feature) > 0:
            extracted_features = torch.cat(L_extracted_feature,dim= -1)
        else:
            extracted_features = None
        return L_node_attributes,extracted_features
    

    def global_spatial_attention(self, 
                                  L_node_attributes: List[torch.Tensor], 
                                  extracted_features: torch.Tensor,
                                    x: torch.Tensor,
                                    contextual: List[torch.Tensor],
                                    L_projected_x: List[torch.Tensor]
                                  )-> List[torch.Tensor]:
        '''
        Compute Spatial Attention between all the spatials units of the target data and the contextual data.
        Spatial Attention has to be computed on Heterogenous spatial units, but can still be used for Homogenous spatial units.

        Args:
        -----
        L_node_attributes : (Optional) =  List of node attributes [B,1,N,L] for each dataset i which need spatial attention
        extracted_features : (Optional) = [B,N,Z] 
        x: input related to target data [B,N,L] 
        contextual: List of tensors related to contextual datasets.

        Returns:
        -------
        For each dataset which need global attention:
        >>> it can be stacked on x (on the channel dimension)
        >>> or it can be concatenated to extracted_features (on the last dimension), in order to be concatenated at the end of the core model.
        '''
        # for ds_name_i,attention_module_i in self.global_s_attn.items():
        for ds_name_i in self.KEY_global_s_attn:
            attention_module_i = self.global_s_attn[ds_name_i]
            #print('Attention Module: ',attention_module_i)
            
            pos_i = self.dict_ds_which_need_attn2pos[ds_name_i] 
            node_attr = contextual[pos_i] 
            # Spatial Attention: 
            # print('node_attr size before attn: ',node_attr.size())
            # print('projected_x size: ',projected_x.size())
            # print('node_attr size: ',node_attr.size())

            if self.dic_stacked_contextual[ds_name_i]:
                projected_x,node_attr = attention_module_i(x,x_contextual = node_attr, dim = 1)   # [B,N,Z*L]
                if self.dic_stacked_consistant_contextual[ds_name_i]:
                    if node_attr.dim() == 3:
                        node_attr = node_attr.unsqueeze(1)
                    if projected_x.dim() == 3:
                        projected_x = projected_x.unsqueeze(1)
                    
                    L_projected_x.append(projected_x)
                else:
                    node_attr = node_attr.reshape(node_attr.size(0),node_attr.size(1),self.L, -1).permute(0, 3, 1, 2) # [B,N,L*Z] -> [B,N,L,Z] -> [B,Z,N,L]

                L_node_attributes.append(node_attr)

            else:
                node_attr = attention_module_i(x,x_contextual = node_attr, dim = 1)   # [B,N,Z*L]
                if extracted_features is not None:
                    extracted_features = torch.cat([extracted_features,node_attr],dim=-1)
                else:
                    extracted_features = node_attr

        return L_node_attributes,L_projected_x,extracted_features
    
    def stack_direct_node_attributes(self, L_node_attributes: List[torch.Tensor], 
                                    contextual: List[torch.Tensor]
                                  )-> List[torch.Tensor]:
        '''
        Some 'Node attribute' doesnot need any spatial attention. 
        It is as example the case for 'subway-out' as contextual data for 'subway-in'. 
        We directly can attribute an attribute  
        '''
        # for pos_i,ds_name_i in self.dict_pos_node_attr2ds.items():
        for pos_i in self.pos_node_attr_which_does_not_need_attn:
            ds_name_i = self.dict_pos_node_attr2ds[pos_i]
            node_attr = contextual[pos_i] 
            if ds_name_i == 'netmob_POIs':
                #permute [B,P,L] ->  [B,L,P]  // reshape : [B,L,P] ->  [B,L,N,C]
                node_attr = node_attr.permute(0, 2, 1)       # [B,L,P]
                added_dim = self.dict_ds2added_dim[ds_name_i]
                node_attr = node_attr.reshape(node_attr.size(0), node_attr.size(1), node_attr.size(2)//added_dim,added_dim).permute(0, 3, 2, 1) # [B,L,N,C]

            if node_attr.dim() == 3:
                node_attr = node_attr.unsqueeze(1)

            L_node_attributes.append(node_attr)
        return L_node_attributes

    def forward(self,x: Tensor,
                contextual: List[Tensor])->Tensor:
        ''' 
        Args:
        -----
        x : 4-th order Tensor: Trafic Flow historical inputs [B,C,N,L]
        contextual : list of contextual data. 
            >>>> contextual[netmob_position]: [B,N,C,H,W,L]
            >>>> contextual[calendar]: [B]
        '''
        L_projected_x: List[torch.Tensor] = torch.jit.annotate(List[torch.Tensor], [])
        B = x.size(0)
        # print('\nx size before forward: ',x.size())
        if self.remove_trafic_inputs:
            #x = torch.Tensor().to(x)
            x = torch.empty(0, device=x.device, dtype=x.dtype)

            # x = x[:, :0].contiguous()

        # print('\nx before stacking new channels:',x.size())
        # print('Contextual size : ',[c_i.size() for c_i in contextual])
        # Spatial Attention and attributing node information: 
        L_node_attributes,extracted_features = self.local_spatial_attention(x,contextual)
        # print('L_node_attributes after local spatial attn: ',[xb.size() for xb in L_node_attributes])
        # print('extracted_features after local spatial attn: ',extracted_features.size() if extracted_features is not None else None)
        L_node_attributes,L_projected_x,extracted_features = self.global_spatial_attention(L_node_attributes,extracted_features,x,contextual,L_projected_x)
        # print('L_node_attributes after global spatial attn: ',[xb.size() for xb in L_node_attributes])
        # print('extracted_features after global spatial attn: ',extracted_features.size() if extracted_features is not None else None)
        L_node_attributes = self.stack_direct_node_attributes(L_node_attributes,contextual)
        # print('L_node_attributes after add_other_node_attributes: ',[xb.size() for xb in L_node_attributes])

        # [B,N,L] -> [B,1,N,L]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        if len(L_node_attributes) > 0:
            # [B,1,N,L] -> [B,C,N,L]
            x = self.stack_node_attributes_from_attn(x,L_node_attributes,L_projected_x)

        """ #A retirer 
        x,extracted_feature = self.forward_feature_extractor_model(x,contextual)        # Tackle NetMob (if exists):
        """


        # print('x after stacking node attribute: ',x.size())
        # print('extracted_feature: ',extracted_features.size() if extracted_features is not None else None)
        x,x_calendar = self.forward_calendar_model(x,contextual)         # Tackle Calendar Data (if exists)

        # print('x after Calendar model: ',x.size())
        #print('CalendarEmbedded Vector: ',time_elt.size() if time_elt is not None else None)

        # Core model 
        """ #A retirer 
        vision_late = self.vision_concatenation_late
        """
        if self.core_model is not None:
            x= self.core_model(x,
                               x_vision = extracted_features,
                               x_calendar = x_calendar,
                                contextual = contextual )
            #print('x after Core Model: ',x.size())
        # ...

        x = reshaping(x,B)
        #print('x after reshaping: ',x.size())
        return(x)
 

def reshaping(x,B):
    if x.dim()==4:
        x = x.squeeze()
        if B == 1:
            x = x.unsqueeze(0)

    # Tackle the case when C=1 and output_dim = 1  
    if x.dim()==2:
        x = x.unsqueeze(-1)

    # Tackle the case when num_nodes = 1
    if x.dim()==1: 
        x = x.unsqueeze(-1)           
        x = x.unsqueeze(-1) 
    return x

def load_model(dataset, args):
    # --- Init L_add and added_dim_input
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
            raise NotImplementedError("This method has not been updated")
    else:
        L_add = 0
        vision_concatenation_late = False
        vision_out_dim = None
    added_dim_input = 0 
    # ---
    for name_i in args.contextual_kwargs.keys():
        if 'out_dim' in args.contextual_kwargs[name_i].keys():
            if 'attn_kwargs' in args.contextual_kwargs[name_i].keys() and 'keep_temporal_dim' in args.contextual_kwargs[name_i]['attn_kwargs'].keys() and args.contextual_kwargs[name_i]['attn_kwargs']['keep_temporal_dim']:
                added_dim_input = added_dim_input +  args.contextual_kwargs[name_i]['out_dim']
            else:
                if ('need_global_attn' in args.contextual_kwargs[name_i].keys() and args.contextual_kwargs[name_i]['need_global_attn']):
                    L_add = L_add + args.contextual_kwargs[name_i]['out_dim']
            
    if 'calendar_embedding' in args.dataset_names: #if not empty 
        # IF Early concatenation : 
        TE_concatenation_late = args.args_embedding.concatenation_late
        TE_embedding_dim = args.args_embedding.embedding_dim
        if args.args_embedding.concatenation_early:
            L_add = L_add + args.args_embedding.embedding_dim
    else:
        TE_concatenation_late = False
        TE_embedding_dim = None

    if args.model_name == 'STAEformer':
        args.added_dim_output = L_add
        args.added_dim_input = added_dim_input
        if TE_concatenation_late or vision_concatenation_late:
            raise NotImplementedError(f'{args.model_name} with TE_concatenation_late has not been implemented')
        filtered_args = {k: v for k, v in vars(args).items() if (k in inspect.signature(STAEformer.__init__).parameters.keys())}
        model = STAEformer(**filtered_args).to(args.device)

    if args.model_name == 'DSTRformer':
        if TE_concatenation_late or vision_concatenation_late:
            raise NotImplementedError(f'{args.model_name} with TE_concatenation_late has not been implemented')
        from pipeline.dl_models.DSTRformer.DSTRformer_utilities import normalize_adj_mx
        filtered_args = {k: v for k, v in vars(args).items() if (k in inspect.signature(DSTRformer.__init__).parameters.keys())}

        adj_mx,_ = load_adj(dataset,adj_type = args.adj_type, threshold=args.threshold)
        adj_mx = normalize_adj_mx(adj_mx, args.adj_normalize_method)
        adj_mx = [torch.tensor(i).to(args.device) for i in adj_mx]
        model = DSTRformer(**filtered_args,adj_mx=adj_mx).to(args.device)


    if args.model_name == 'STGformer':
        if TE_concatenation_late or vision_concatenation_late:
            raise NotImplementedError(f'{args.model_name} with TE_concatenation_late has not been implemented')
        from pipeline.dl_models.STGformer.STGformer_utilities import normalize_adj_mx
        filtered_args = {k: v for k, v in vars(args).items() if (k in inspect.signature(STGformer.__init__).parameters.keys())}

        # Useless in this version :
        if hasattr(args,'adj_type'): 
            raise NotImplementedError('STGformer with adjacency matrix has not been implemented yet')
            adj_mx,_ = load_adj(dataset,adj_type = args.adj_type, threshold= args.threshold)
            adj_mx = normalize_adj_mx(adj_mx, args.adj_normalize_method, return_type="dense")
            supports = [torch.tensor(i).to(args.device) for i in adj_mx]
            model = STGformer(**filtered_args,supports=supports).to(args.device)
        # ...
        model = STGformer(**filtered_args).to(args.device)
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
        adj,_ = load_adj(dataset,adj_type = args.adj_type, threshold= args.threshold)
        model = DCRNN(adj, **vars(args)).to(args.device)
        
    if args.model_name == 'STGCN':
        from pipeline.dl_models.STGCN.get_gso import get_output_kernel_size, get_block_dims, get_gso_from_adj
        former_L = None
        for key in args.contextual_kwargs.keys():
            if 'stack_consistent_datasets' in args.contextual_kwargs[key].keys() and args.contextual_kwargs[key]['stack_consistent_datasets']:
                    # Specific case when we do 'attention late':
                    if not(('attn_late' in args.contextual_kwargs[key]['attn_kwargs']) and  (args.contextual_kwargs[key]['attn_kwargs']['attn_late'])):
                        L_new = args.contextual_kwargs[key]['attn_kwargs']['dim_model']
                        if former_L is not None:
                            if former_L != L_new:
                                raise ValueError(f'Inconsistent L_add_2 for {key} contextual dataset: {former_L} != {L_new}')
                        else: 
                            former_L = L_new

        args.L = former_L if former_L is not None else args.L
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
        from pipeline.dl_models.MLP.MLP import MLP_output 
        print('\n>>>>Model == MLP, keep in mind Concatenation Late DOES NOT WORK here. Only Concatenation Early')
        print(f'>>>>Also Stupid model. Input_dim = h_dim = L+L_add. Output_dim = {args.out_dim}')
        input_dim = args.L+L_add
        model = MLP_output(input_dim=input_dim,out_h_dim=input_dim,num_nodes=None,embedding_dim=args.out_dim,multi_embedding=False,dropout=args.dropout).to(args.device)

    if args.model_name == 'ARIMA':
        # on instancie directement le module SARIMAXModule
        model = SARIMAX(
            order=args.order,
            seasonal_order=args.seasonal_order,
            enforce_stationarity=args.enforce_stationarity,
            enforce_invertibility=args.enforce_invertibility
        ).to(args.device)

    if args.model_name == 'XgBoost':
        # on instancie directement le module XGBoostModule
        model = XGBoost(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            gamma=args.gamma,
            # learning_rate=args.learning_rate,
            # reg_alpha=args.reg_alpha,
            # reg_lambda=args.reg_lambda,
            # objective=args.objective,
            # eval_metric=args.eval_metric
        ).to(args.device)

    if args.model_name == None:
        model = None
    return(model,args)