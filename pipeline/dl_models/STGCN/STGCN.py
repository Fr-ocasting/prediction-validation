from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn

# Relative path:
import sys 
import os 
import importlib 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal import:
import pipeline.dl_models.STGCN.STGCN_layer as layers
from pipeline.dl_models.MTGNN.MTGNN_layer import graph_constructor
from pipeline.utils.utilities import filter_args


# ============================================================
# Inspired by  https://github.com/hazdzz/STGCN/tree/main
# ============================================================

#  -- STGCN Layer x L -- > output Module -- > Prediction 
#
# output Module =  Convolution (Pooling sur un axe) -- FC1 -- ReLU -- FC2 

class STGCN(nn.Module):
    # STGCN contains 'TGTND TGTND TNFF' structure

    # GraphConv is the graph convolution from GCN.
    # GraphConv is not the first-order ChebConv, because the renormalization trick is adopted.

    # ChebGraphConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind as a graph filter.
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, args, gso, blocks,Ko,backbone = False):
        # super(STGCN, self).__init__()
        super().__init__()

        self.out_dim = blocks[-1][-1]
        modules = []
        self.init_learnable_adjacency_matrix(args.learnable_adj_matrix,
                                             args.num_nodes,
                                             k=args.learnable_adj_top_k if getattr(args,'learnable_adj_matrix') else None,
                                             node_embedding_dim=args.learnable_adj_embd_dim if getattr(args,'learnable_adj_matrix') else None,
                                             device = args.device,
                                             alpha=3)
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, args.num_nodes, blocks[l][-1], blocks[l+1], args.act_func, args.graph_conv_type, gso, args.enable_bias, args.dropout,args.enable_padding,self.g_constructor))
        self.st_blocks = nn.Sequential(*modules)

        # ---- Tackle Input if 'extracted feature' is concatenated Late:  -----
        # self.concatenation_late = args.args_vision.concatenation_late if (hasattr(args,'args_vision') and hasattr(args.args_vision,'concatenation_late'))else False
        self.concatenation_late = False
        extracted_feature_dim = 0
        print(' \n------------\nStart Counting Extracted Feature Dim')
        if (hasattr(args,'contextual_kwargs')):
            for key in args.contextual_kwargs.keys():
                # print('   key : ',key)
                # If this data does not have an attention module which is concatenated late:
                if not('attn_kwargs' in args.contextual_kwargs[key].keys() and 'attn_late' in args.contextual_kwargs[key]['attn_kwargs'].keys() and  args.contextual_kwargs[key]['attn_kwargs']['attn_late']):
                    local_concatenation_late = not(args.contextual_kwargs[key]['stacked_contextual']) 
                    self.concatenation_late = self.concatenation_late or local_concatenation_late
                    # print('      local_concatenation_late: ',local_concatenation_late)
                    print('      concatenation_late: ',self.concatenation_late)
                    if local_concatenation_late:
                        print("      extracted_feature_dim += ", args.contextual_kwargs[key]['out_dim'])
                        extracted_feature_dim = extracted_feature_dim + args.contextual_kwargs[key]['out_dim'] #* args.contextual_kwargs[key]['attn_kwargs']['L_out']
        print('Total Extracted Feature Dim: ',extracted_feature_dim)
        print('------------\n')

        self.TE_concatenation_late = args.args_embedding.concatenation_late if 'calendar_embedding' in args.dataset_names else False 
        # ---- 

        self.backbone = backbone
        self.Ko = Ko
        self.num_nodes = args.num_nodes
            

        if hasattr(args.args_embedding,'embedding_dim'):
            embedding_dim = args.args_embedding.embedding_dim 
        else:
            embedding_dim = None


        in_feature_fc1 = blocks[-3][-1] 

        # Init for torch compile
        self.output = nn.Identity()
        self.fc1 = nn.Identity()
        self.fc2 = nn.Identity()
        self.relu = nn.ReLU()



        # ----- Tackle Spatial Attention Late -----
        ModuleContextualAttnLate = nn.ModuleDict() # For the late concatenation of contextual attention modules
        attn_late_dim = 0
        for ds_name in args.contextual_kwargs.keys():
            if 'attn_kwargs' in args.contextual_kwargs[ds_name].keys() and 'attn_late' in args.contextual_kwargs[ds_name]['attn_kwargs'].keys():
                if args.contextual_kwargs[ds_name]['attn_kwargs']['attn_late']:
                    script = importlib.import_module(f"pipeline.dl_models.SpatialAttn.SpatialAttn")
                    scrip_args = importlib.import_module(f"pipeline.dl_models.SpatialAttn.load_config")
                    importlib.reload(scrip_args)    
                    args_ds_i = scrip_args.args
                    args_ds_i.dropout = args.dropout        
 
                    args_ds_i.key_dim = args.L  # input dim of Key  --> L 
                    args_ds_i.output_temporal_dim = args.contextual_kwargs[ds_name]['attn_kwargs']['L_out'] if 'L_out' in args.contextual_kwargs[ds_name]['attn_kwargs'].keys() else None   #  Dimension of FC layer after MHA ONLY IF stack_consistent_datasets is False  & if we set L_out.
                    args_ds_i.stack_consistent_datasets = args.contextual_kwargs[ds_name]['stack_consistent_datasets'] if 'stack_consistent_datasets' in args.contextual_kwargs[ds_name].keys() else False # True if we want the output of MHA, False if we want it to also pass through FC layer after MHA  
                    args_ds_i.proj_query = True if 'proj_query' not in args.contextual_kwargs[ds_name]['attn_kwargs'].keys() else args.contextual_kwargs[ds_name]['attn_kwargs']['proj_query']
                    for key,value in args.contextual_kwargs[ds_name]['attn_kwargs'].items():
                        setattr(args_ds_i,key,value)
                    args_ds_i.dim_model = blocks[-3][-1] # Dimension of the last STGCN block output channel
                    args_ds_i.query_dim = args_ds_i.dim_model  # input dim of Query  --> output dim of STGCN 
                    importlib.reload(script)
                    func = script.model
                    filered_args = filter_args(func, args_ds_i)
                    ModuleContextualAttnLate[ds_name] = func(**filered_args)      
                    if args_ds_i.stack_consistent_datasets:
                        attn_late_dim = attn_late_dim+args_ds_i.dim_model
                    else:
                        attn_late_dim = (attn_late_dim
                                        + (args.contextual_kwargs[ds_name]['attn_kwargs']['input_embedding_dim'])
                                        + (args.contextual_kwargs[ds_name]['attn_kwargs']['adaptive_embedding_dim'] if 'adaptive_embedding_dim' in args.contextual_kwargs[ds_name]['attn_kwargs'].keys() else 0)
                                        + (args.contextual_kwargs[ds_name]['attn_kwargs']['tod_embedding_dim'] if 'tod_embedding_dim' in args.contextual_kwargs[ds_name]['attn_kwargs'].keys() else 0)
                                        + (args.contextual_kwargs[ds_name]['attn_kwargs']['dow_embedding_dim'] if 'dow_embedding_dim' in args.contextual_kwargs[ds_name]['attn_kwargs'].keys() else 0)
                                    )
        # -----

        # ---- Added dim output:
        added_dim_output = 0
        for name_i in args.contextual_kwargs.keys():
            if 'attn_kwargs' in args.contextual_kwargs[name_i].keys() and 'keep_temporal_dim' in args.contextual_kwargs[name_i]['attn_kwargs'].keys() and args.contextual_kwargs[name_i]['attn_kwargs']['keep_temporal_dim']:
                if 'concatenation_late' in args.contextual_kwargs[name_i]['attn_kwargs'].keys() and args.contextual_kwargs[name_i]['attn_kwargs']['concatenation_late']:
                    added_dim_output = added_dim_output + args.contextual_kwargs[name_i]['out_dim']
                    
            else:
                if ('need_global_attn' in args.contextual_kwargs[name_i].keys() and args.contextual_kwargs[name_i]['need_global_attn']):
                    added_dim_output = added_dim_output + args.contextual_kwargs[name_i]['out_dim']

        if self.backbone: 
            self.temporal_agg = layers.TemporalConvLayer(Ko, 
                                                  c_in = in_feature_fc1, 
                                                  c_out = blocks[-2][0], 
                                                  num_nodes = args.num_nodes, 
                                                  act_func= args.act_func,
                                                  enable_padding = False)
            self.tc1_ln = nn.LayerNorm([args.num_nodes, blocks[-2][0]])
        else:
            if self.Ko > 0:
                #print('blocks: ',blocks)
                #print('in_feature_fc1: ',in_feature_fc1)
                self.output = layers.OutputBlock(self.Ko, in_feature_fc1, blocks[-2], blocks[-1][0], args.num_nodes, args.act_func, args.enable_bias, args.dropout,
                                                self.concatenation_late,extracted_feature_dim,
                                                self.TE_concatenation_late,embedding_dim,args.temporal_graph_transformer_encoder,
                                                TGE_num_layers=args.TGE_num_layers if args.temporal_graph_transformer_encoder else None, 
                                                TGE_num_heads=args.TGE_num_heads if args.temporal_graph_transformer_encoder else None,  
                                                TGE_FC_hdim=args.TGE_FC_hdim if args.temporal_graph_transformer_encoder else None, 
                                                ModuleContextualAttnLate = ModuleContextualAttnLate,
                                                dict_ds_which_need_attn_late2pos = args.dict_ds_which_need_attn_late2pos,
                                                attn_late_dim = attn_late_dim,
                                                added_dim_output = added_dim_output
                                                )
            elif self.Ko == 0:
                self.fc1 = nn.Linear(in_features=in_feature_fc1, out_features=blocks[-2][0], bias=args.enable_bias)
                self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
        

        # # --- Tackles x_vision_early, x_vision_late :
        # for ds_name, kwargs_i in self.contextual_kwargs.items():
        #     if 'emb_dim' in kwargs_i.keys():
        #         # Temporal Proj: 
        #         self.contextual_emb[ds_name] = TEMPORAL CONV#(kwargs_i['C'], kwargs_i['emb_dim'])

        #         # Spatial Proj (or Repeat):
        #         if (
        #             'n_spatial_unit' in kwargs_i.keys()
        #             and kwargs_i['n_spatial_unit'] is not None
        #             and (
        #                 'repeat_spatial' not in kwargs_i.keys()
        #                 or not kwargs_i['repeat_spatial']
        #                 )
        #             and (
        #                 'spatial_proj' not in kwargs_i.keys()
        #                 or kwargs_i['spatial_proj']
        #             )
        #         ):
        #             self.contextual_spatial_proj[ds_name] = nn.Linear(kwargs_i['n_spatial_unit'], self.num_nodes)
        # # ----

        
    def init_learnable_adjacency_matrix(self,bool_learnable_adj,num_nodes,k,node_embedding_dim,device,alpha):
        if bool_learnable_adj:
            self.g_constructor = graph_constructor(num_nodes, k, node_embedding_dim, device=device, alpha=alpha, static_feat=None).to(device)     
        else:
            self.g_constructor = None


    def forward(self,x: Tensor,
                x_vision_early: Optional[Tensor] = None,
                x_vision_late: Optional[Tensor] = None,
                x_calendar: Optional[Tensor] = None,
                contextual: Optional[list[Tensor]]= None) -> Tensor:
        
        ''' 
        Args:
        -------
        x: 3-th or 4-th order Tensor : [B,N,L] or [B,C,N,L]

            B: batch-size
            C: number of traffic channel (flow, speed, density ...)
            N: number of spatial-units (exemple: 40 subway stations)
            L: length of historical sequence (t-w,t-d,t-6,t-5,t-4,t-3,t-2,t-1)

        1st step: reshape permute input for first st_blocks : [B,C,L,N] 
        
        '''
        if x_vision_early is not None:
            raise NotImplementedError('x_vision has not been implemented')
        # print('\nx.size: ',x.size())
        # Tackle case where we only want to use the output module (and not the core-model STGCN
        if not (x.numel() == 0):
            # Reshape and permute : [B,N,L] or [B,C,N,L] ->  [B,C,L,N]
            if len(x.size())<4:
                x = x.unsqueeze(1)
            ### Core model :

            if not x.numel() == 0:
                #[B,C,N,L] -> [B,C,L,N]
                x = x.permute(0,1,3,2)
                # [B,C,L,N] -> [B, C_out, L-4*nb_blocks, N]
                x = self.st_blocks(x)
            ### ---
        if self.backbone:
            x = self.temporal_agg(x)
            x = self.tc1_ln(x.permute(0, 2, 3, 1)) 
            return x

        else:
        # --- Output Module :
            if self.Ko >= 1:
                # Causal_TempConv2D - FC(128,128) -- FC(128,1) -- LN - ReLU --> [B,1,1,N]
                x = self.output(x,x_vision_late,x_calendar,contextual)
            elif self.Ko == 0:
                # [B,C_out,L',N] = [B,1,L',N] actually 
                if x_vision_late is not None:
                    # [B,C_out,N,L'] -> [B,C_out,L',N] 
                    x_vision_late = x_vision_late.permute(0,1,3,2)
                    # Concat [B,C,L-4*nb_blocks, N] + [B,C_out,L',N]
                    if not (x.numel() == 0):
                        x = torch.cat([x,x_vision_late],dim=2)
                    else:
                        x = x_vision_late
                if self.TE_concatenation_late and x_calendar is not None:
                    # [B,C,N,L_calendar]  -> [B,C,L_calendar,N] 
                    x_calendar = x_calendar.permute(0,1,3,2)
                    # Concat [B,C,L-4*nb_blocks, N] + [B,C,L_calendar,N] 
                    if not (x.numel() == 0):
                        x = torch.cat([x,x_calendar],dim=2)
                    else:
                        x = x_calendar

        
            x = self.fc1(x.permute(0, 2, 3, 1))

            x = self.relu(x)
            x = self.fc2(x)
            x = x.permute(0, 3, 1, 2)


            
            # print('x.size: ' ,x.size())
            B = x.size(0)
            x = x.squeeze()
            # print('x.size: ' ,x.size())
            # print('self.num_nodes: ', self.num_nodes)
            # print('self.out_dim: ', self.out_dim)
            if B ==1:
                x = x.unsqueeze(0)
            if self.num_nodes == 1:
                x = x.unsqueeze(-1)
            if self.out_dim == 1:
                x = x.unsqueeze(-2)
            # print('x.size: ' ,x.size())

            x = x.permute(0,2,1)

            return x    