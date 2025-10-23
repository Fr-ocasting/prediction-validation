"""
STAEformer from  https://github.com/xdzhelheim/staeformer
"""
from torch import Tensor
import torch.nn as nn
import torch
from typing import Optional,List


def repeat_transpose(x: Tensor, n_repeat: int) -> Tensor:
    if x.size(1) == 1:
        x = x.repeat(1, n_repeat, 1, 1)  # [B,1,L,emb_dim] -> [B,N,L,emb_dim]
        x = x.transpose(1, 2)  # [B,N,L,emb_dim] -> [B,L,N,emb_dim]
    return x


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False,KV_model_dim = None):
        super().__init__()

        self.model_dim = model_dim
        self.KV_model_dim = KV_model_dim if KV_model_dim is not None else model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads
        assert ( self.head_dim * num_heads == model_dim), f"model_dim {model_dim} must be divisible by num_heads {num_heads}"

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(self.KV_model_dim, model_dim)
        self.FC_V = nn.Linear(self.KV_model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        # print('\nquery size:', query.size())
        # print('key size:', key.size())
        # print('value size:', value.size())

        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)
        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        self.attn_score = attn_score

        # print('attn_score size:', attn_score.size())
        # print('value size:', value.size())
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)
        # print('out size before out_proj:', out.size())
        out = self.out_proj(out)
        # print('out size after out_proj:', out.size())
        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False,KV_model_dim = None
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask,KV_model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, dim: int = -2,x_contextual: Tensor = None) -> Tensor:
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        if x_contextual is not None:
            x_contextual = x_contextual.transpose(dim, -2)
            out = self.attn(x, x_contextual, x_contextual)  # (batch_size, ..., length, model_dim)
        else:
            out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        # print('context: ',out.size())
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out

        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        # print('feed forward: ',out.size())
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        # print('output: ',out.size())
        return out


class ContextualInputEmbedding(nn.Module):
    """
    Contextual features embedding + Keep track on extracted feature for early and late fusion.
    """
    def __init__(
        self,
        num_nodes,
        contextual_kwargs = {},
        sum_contextual_dim = 0,
        contextual_positions = {},
        Early_fusion_names = [],
        Late_fusion_names = [],
    ):

        super().__init__()
        self.num_nodes: int = num_nodes
        self.sum_contextual_dim = sum_contextual_dim
        self.contextual_kwargs = contextual_kwargs
        self.contextual_positions = contextual_positions
        self.contextual_emb = nn.ModuleDict()
        self.contextual_spatial_proj = nn.ModuleDict()
        self.Early_fusion_names = Early_fusion_names
        self.Late_fusion_names = Late_fusion_names
        self.projected_contextual = {}
        # --- Define Temporal and Spatial Embedding (or Spatial Repeat) layers:
        for ds_name, kwargs_i in self.contextual_kwargs.items():
            if 'emb_dim' in kwargs_i.keys():
                # Temporal Proj 
                self.contextual_emb[ds_name] = nn.Linear(kwargs_i['C'], kwargs_i['emb_dim'])

                # Spatial Proj (or Repeat):
                if (
                    'n_spatial_unit' in kwargs_i.keys()
                    and kwargs_i['n_spatial_unit'] is not None
                    and (
                        'repeat_spatial' not in kwargs_i.keys()
                        or not kwargs_i['repeat_spatial']
                        )
                    and (
                        'spatial_proj' not in kwargs_i.keys()
                        or kwargs_i['spatial_proj']
                    )
                ):
                    self.contextual_spatial_proj[ds_name] = nn.Linear(kwargs_i['n_spatial_unit'], self.num_nodes)
        # ---


    def forward(self, 
                contextual: Optional[list[Tensor]]= None,
                ) -> Tensor:
        
        # Temporal Embedding
        self.t_embedding(contextual)

        # Spatial Embedding (or Repeat) if needed :
        self.s_embedding(n_repeat = self.num_nodes)

        early_features = [contextual_i for ds_name, contextual_i in self.projected_contextual.items() if ds_name in self.Early_fusion_names]
        self.early_features = torch.cat(early_features, dim=-1) if len(early_features)>0 else torch.empty(0)

        late_features = [contextual_i for ds_name, contextual_i in self.projected_contextual.items() if ds_name in self.Late_fusion_names]
        self.late_features = torch.cat(late_features, dim=-1) if len(late_features)>0 else torch.empty(0)


    def t_embedding(self,contextual):
        for ds_name, _ in self.contextual_emb.items():
            # print('    ds_name: ',ds_name, '(','Late fusion' if ds_name in self.Late_fusion_names else 'Early fusion' if ds_name in self.Early_fusion_names else 'No fusion',')')
            contextual_i = contextual[self.contextual_positions[ds_name]] 
            # print(f'        contextual shape before embedding:', contextual_i.size())
            # Align the dimensions
            if contextual_i.dim() ==3:
                contextual_i = contextual_i.unsqueeze(-1) # [B,P,L] -> [B,P,L,1]
            # Embedding on channel dim: 
            contextual_i = self.contextual_emb[ds_name](contextual_i)  # [B,P,L,C] -> [B,P,L,emb_dim]
            # print('        Temporal proj:', contextual_i.size())

            self.projected_contextual[ds_name] = contextual_i

    def s_embedding(self,n_repeat):
        # if Need spatial projection : 
        for ds_name, _ in self.contextual_emb.items():
            contextual_i = self.projected_contextual[ds_name]

            if ds_name in self.contextual_spatial_proj.keys():
                contextual_i = self.contextual_spatial_proj[ds_name](contextual_i.permute(0,3,2,1))  # [B,P,L,emb_dim] -permute-> [B,emb_dim,L,P] -> [B,emb_dim,L,N]
                contextual_i = contextual_i.permute(0,2,3,1)  # [B,emb_dim,L,N] -> [B,L,N,emb_dim]
                # print('        Spatial projection: ', contextual_i.size())

            # If not projected and need to repeat on spatial dim:  
            contextual_i = repeat_transpose(contextual_i, n_repeat)
            

            self.projected_contextual[ds_name] = contextual_i

    def concat_features(self, 
                        x: Tensor, 
                        features: Tensor) -> Tensor:
        if features.size(-1) == 0:
            return x
        else: 
            x = torch.cat([x, features], dim=-1)
            return x


class STAEformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        L=12,
        step_ahead=12,
        time_step_per_hour=12,
        C=1,
        out_dim_factor = 1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
        contextual_positions = {},
        horizon_step = 1,
        added_dim_output = 0,
        added_dim_input = 0,
        contextual_kwargs = {},
        Early_fusion_names = [],
        Late_fusion_names = [],
    ):
        super().__init__()
        #  ---  self attributes: ---
        # self.concatenation_late = added_dim_output > 0
        self.num_nodes: int = num_nodes
        self.in_steps = L
        self.out_steps = step_ahead
        self.steps_per_day = 24*time_step_per_hour
        self.input_dim = C
        self.output_dim = out_dim_factor
        self.horizon_step = horizon_step 
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.added_dim_output = added_dim_output
        self.added_dim_input = added_dim_input

        self.sum_contextual_dim = 0
        for k in contextual_kwargs.keys():
            concatenation_late_bool = ('attn_kwargs' in contextual_kwargs[k].keys()) and ('concatenation_late' in contextual_kwargs[k]['attn_kwargs'].keys()) and (contextual_kwargs[k]['attn_kwargs']['concatenation_late'])
            # Embedding with concatenation early : 
            if 'emb_dim' in contextual_kwargs[k].keys() and not( concatenation_late_bool ):
                self.sum_contextual_dim = self.sum_contextual_dim + contextual_kwargs[k]['emb_dim']
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
            + added_dim_input
            + self.sum_contextual_dim
        )
        self.output_model_dim = self.model_dim + self.added_dim_output
        self.Early_fusion_names = Early_fusion_names
        self.Late_fusion_names = Late_fusion_names

        # print('input embedding dim:', self.input_embedding_dim)
        # print('tod_embedding_dim:', self.tod_embedding_dim)
        # print('dow_embedding_dim:', self.dow_embedding_dim)
        # print('spatial_embedding_dim:', self.spatial_embedding_dim)
        # print('adaptive_embedding_dim:', self.adaptive_embedding_dim)
        # print('added_dim_input:', self.added_dim_input)
        # print('added_dim_output:', self.added_dim_output)
        # print('sum_contextual_dim:', self.sum_contextual_dim)
        # print('Total model_dim:', self.model_dim)
        # print('Total output_model_dim:', self.output_model_dim)



        
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(self.input_dim, input_embedding_dim)
        #  --- 

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(self.steps_per_day, tod_embedding_dim)
        else:
            self.tod_embedding = None

        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        else:
            self.dow_embedding = None
        
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        else:
            self.node_emb = None

        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                              nn.Parameter(
                 torch.empty(self.in_steps, self.num_nodes, adaptive_embedding_dim))
            )
        else:
            self.adaptive_embedding = None
            
        self.contextual_kwargs = contextual_kwargs
        self.contextual_positions = contextual_positions
        self.pos_tod = contextual_positions.get("calendar_timeofday", None)
        self.pos_dow = contextual_positions.get("calendar_dayofweek", None)
        self.contextual_input_embedding =  ContextualInputEmbedding(self.num_nodes,self.contextual_kwargs,
                                                                    self.sum_contextual_dim,self.contextual_positions,
                                                                    self.Early_fusion_names,self.Late_fusion_names)

        if use_mixed_proj:
            self.output_proj = nn.Linear(self.in_steps * self.output_model_dim, self.out_steps * self.output_dim// self.horizon_step)
            self.temporal_proj = None
        else:
            self.temporal_proj = nn.Linear(self.in_steps, self.out_steps// self.horizon_step)
            self.output_proj = nn.Linear(self.output_model_dim, self.out_steps * self.output_dim// self.horizon_step)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        
    def forward(self,  x: Tensor,
                # x_vision: Optional[Tensor] = None, 
                x_vision_early: Optional[Tensor] = None,
                x_vision_late: Optional[Tensor] = None,
                x_calendar: Optional[Tensor] = None,
                contextual: Optional[list[Tensor]]= None,
                ) -> Tensor:
        
        # print('\n--------------------------------------------\nx shape before input_proj:', x.shape)

        if x_calendar is None:
            raise ValueError("x_calendar is None. Set args.calendar_types to ['dayofweek', 'timeofday'] and add 'calendar' to dataset_names.")

        else: 
            if x_calendar.dim() ==3:
                x_calendar = x_calendar.unsqueeze(2)  # [B,L,2] -> [B,L,1,2]
            x = x.permute(0,3,2,1) # [B,C,N,L] -> [B,L,N,C]
            if x_calendar.size(-1) != 2:
                raise ValueError(f"Expected x_calendar.size(-1) == 2, but got {x_calendar.size(-1)}. Set args.calendar_types to ['dayofweek', 'timeofday'] and add 'calendar' to dataset_names.")
            x_calendar = x_calendar.repeat(1,1,self.num_nodes,1) # [B,L,1,2]-> [B,L,N,2]

            # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
            batch_size = x.shape[0]
            x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
            # print('   x size after input_proj:', x.size())
            # print(    'x_calendar shape:', x_calendar.shape)
            
            # Init features with empty tensor with 0 channels
            features = torch.empty(x.size(0), x.size(1), x.size(2), 0,dtype=x.dtype, device=x.device)  

            #print('features shape:', features.shape)
            if self.tod_embedding is not None:
                # print('   add TOD embedding')
                tod = x_calendar[..., self.pos_tod]
                # print('      tod shape before embedding:', tod.size())
                tod_emb = self.tod_embedding( (tod * self.steps_per_day).long() )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)

                # print('      tod_emb shape after embedding:', tod_emb.size())
                # print('      features shape before adding tod_emb:', features.size())
                features = torch.cat([features, tod_emb], dim=-1)

            if self.dow_embedding is not None:
                # print('   add DOW embedding')
                dow = x_calendar[..., self.pos_dow]
                dow_emb = self.dow_embedding(dow.long())  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
                features = torch.cat([features, dow_emb], dim=-1)

            if self.node_emb is not None:
                spatial_emb = self.node_emb.expand(batch_size, self.in_steps, self.num_nodes, self.spatial_embedding_dim)
                features = torch.cat([features, spatial_emb],dim= -1)

            if self.adaptive_embedding is not None:
                # print('   add adpt embedding')
                adp_emb = self.adaptive_embedding.expand(size=(batch_size, self.in_steps, self.num_nodes, self.adaptive_embedding_dim))
                features = torch.cat([features, adp_emb], dim=-1)
            
            # Add contextual features as inputs (Embedding + concatenation early)
            # print('   features size before  Early Fusion of Embedding of Contextual: ',features.size())
            self.contextual_input_embedding(contextual)
            features = self.contextual_input_embedding.concat_features(features, self.contextual_input_embedding.early_features)
            # print('   features size after Early Fusion of Embedding of Contextual: ',features.size())


            if x_vision_early is not None: 
                # print('   concat x_vision early')
                features = torch.cat([features, x_vision_early], dim=-1)
                # print('      features size after concat early of feature extracted from contextual : ',features.size())



            x = torch.cat([x, features], dim=-1)
        
            # print('   x size after adding all embeddings:', x.size())


            for attn in self.attn_layers_t:
                # print('\n--- Temporal attention layer ---')
                x = attn(x, dim=1)
            for attn in self.attn_layers_s:
                # print('\n--- Spatial attention layer ---')
                x = attn(x, dim=2)
            # (batch_size, in_steps, num_nodes, model_dim)
            # print('\nx.size() after T-attn and S-attn : ',x.size())

            # If Fusion Late of embedding : 
            # print('   x.size() before Late fusion of Embedding : ',x.size())
            x = self.contextual_input_embedding.concat_features(x, self.contextual_input_embedding.late_features)
            # print('   x.size() after Late fusion of Embedding : ',x.size())

            if x_vision_late is not None: 
                # print('   concat x_vision late')
                # print('      x.size() before concat x_vision_late: ',x.size())
                # print('      x_vision_late.size(): ',x_vision_late.size())
                x = torch.cat([x, x_vision_late], dim=-1)

            if self.temporal_proj is None:
                out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
                # print('\out size before mixed proj:', out.size())
                # print(f"    Number of each elmt in 'out': , {out.numel()/batch_size}")
                # print('   self.num_nodes:',self.num_nodes)
                # print('   self.in_steps:',self.in_steps)
                # print('   self.model_dim:',self.model_dim)
                # print('   self.added_dim_output:',self.added_dim_output)
                # print('   Expected reshaped last dim: in_steps *(model_dim + added_dim_output) = ', self.in_steps, 'x' ,(self.model_dim+self.added_dim_output),'=', self.in_steps * (self.model_dim+self.added_dim_output))
                # print('   self.output_model_dim:',self.output_model_dim)
                # print('   output_proj:',self.output_proj)

                out = out.reshape( batch_size, self.num_nodes, self.in_steps * (self.model_dim+self.added_dim_output))
                out = self.output_proj(out)
                out = out.view(batch_size, self.num_nodes, self.out_steps//self.horizon_step, self.output_dim)
                out = out.transpose(1, 2)  # (batch_size, self.out_steps//self.horizon_step, num_nodes, output_dim)
            else:
                out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
                out = self.temporal_proj(out)  # (batch_size, model_dim, num_nodes, self.out_steps//self.horizon_step)
                out = self.output_proj(out.transpose(1, 3) )  # (batch_size, self.out_steps//self.horizon_step, num_nodes, output_dim)
            
            out = out.permute(0,3,2,1)  #  (batch_size, output_dim, num_nodes, self.out_steps//self.horizon_step )
            # print('output.size(): ',out.size())

            return out



class MultiLayerCrossAttention(nn.Module):
    def __init__(
        self, input_embedding_dim, feed_forward_dim=2048, num_heads=8,num_layers = 3, 
        dropout=0, 
        mask=False,
        steps_per_day = 288,
        tod_embedding_dim=0,
        dow_embedding_dim=0,
        pos_tod = None,
        pos_dow = None,
        in_steps = 7,
        Q_num_nodes = 40,
        KV_num_nodes = 13,
        init_adaptive_query_dim = 0 ,
        adaptive_embedding_dim = 0, 
    ):
        super().__init__()
        self.contextual_proj = nn.Linear(1, input_embedding_dim)
        self.model_dim = input_embedding_dim+tod_embedding_dim+dow_embedding_dim+adaptive_embedding_dim
        self.attn_layers = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.steps_per_day = steps_per_day
        self.in_steps = in_steps
        self.Q_num_nodes = Q_num_nodes 
        self.KV_num_nodes = KV_num_nodes
        self.init_adaptive_query_dim = init_adaptive_query_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        
        if self.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
            self.pos_tod = pos_tod
        else:
            self.tod_embedding = None

        if self.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
            self.pos_dow = pos_dow
        else:
            self.dow_embedding = None

        # -- Query Initialization : choose between adaptive tensor or input x
        if self.init_adaptive_query_dim >0:
            self.init_adaptive_query = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.in_steps, self.Q_num_nodes, self.init_adaptive_query_dim)))
            self.input_proj = None
 
        else:
            self.init_adaptive_query = None
            self.input_proj = nn.Linear(1, input_embedding_dim) 
        # ----

        # -- Adaptive Embedding for added to Query and Key,Value
        if adaptive_embedding_dim > 0:
            self.Q_adaptive_embedding = nn.init.xavier_uniform_(
                              nn.Parameter(torch.empty(self.in_steps, self.Q_num_nodes, adaptive_embedding_dim)))
            self.KV_adaptive_embedding = nn.init.xavier_uniform_(
                              nn.Parameter(torch.empty(self.in_steps, self.KV_num_nodes, adaptive_embedding_dim)))
        else:
            self.Q_adaptive_embedding = None
            self.KV_adaptive_embedding = None
        # ----


    def forward(self, x: Tensor, x_contextual: Tensor = None,x_calendar : Tensor = None,dim: int = -2) -> Tensor:

        batch_size = x.size(0)

        # Use Adapive Tensor as Query : 
        if self.init_adaptive_query is not None:
            # print('generate Query from adaptive tensor')
            query_init = self.init_adaptive_query.expand(size=(batch_size, self.in_steps, self.Q_num_nodes, self.init_adaptive_query_dim))
        # ...
        
        # Use X as query : 
        else:
            # print('init x.size(): ',x.size())
            if x.dim()==3:
                x = x.unsqueeze(-1)
            # print('after unsqueeze: ',x.size())
            x = self.input_proj(x)
            x = x.transpose(1,2)  
            # print('x.size() after input proj and transpose: ',x.size())
            query_init  = x
        # ...

        if x_contextual is not None:
            # print('init x_contextual.size(): ',x_contextual.size())
            if x_contextual.dim()==3:
                x_contextual = x_contextual.unsqueeze(-1)
            # print('after unsqueeze: ',x.size())
            x_contextual = self.contextual_proj(x_contextual)
            x_contextual  = x_contextual.transpose(1,2) if x_contextual is not None else None
            # print('x_contextual.size() after contextual proj and transpose: ',x.size())
        

        calendar_features = torch.empty(query_init.size(0), query_init.size(1),1, 0,dtype=query_init.dtype, device=query_init.device)
        if x_calendar is not None:
            if x_calendar.size(-1) != 2:
                    raise ValueError(f"Expected x_calendar.size(-1) == 2, but got {x_calendar.size(-1)}. Set args.calendar_types to ['dayofweek', 'timeofday'] and add 'calendar' to dataset_names.")
            if x_calendar.dim() ==3:
                    x_calendar = x_calendar.unsqueeze(2)  # [B,L,2] -> [B,L,1,2]
            if self.tod_embedding is not None:
                tod = x_calendar[..., self.pos_tod]
                tod_emb = self.tod_embedding( (tod * self.steps_per_day).long() )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
                calendar_features = torch.cat([calendar_features, tod_emb], dim=-1)

            if self.dow_embedding is not None:
                dow = x_calendar[..., self.pos_dow]
                dow_emb = self.dow_embedding(dow.long())  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
                calendar_features = torch.cat([calendar_features, dow_emb], dim=-1)
        
            query = torch.cat([query_init, calendar_features.repeat(1,1,query_init.size(2),1)], dim=-1)
            x_contextual = torch.cat([x_contextual, calendar_features.repeat(1,1,x_contextual.size(2),1)], dim=-1)


        if self.Q_adaptive_embedding is not None:
            Q_adp_emb = self.Q_adaptive_embedding.expand(size=(batch_size, self.in_steps, self.Q_num_nodes, self.adaptive_embedding_dim))
            KV_adp_emb = self.KV_adaptive_embedding.expand(size=(batch_size, self.in_steps, self.KV_num_nodes, self.adaptive_embedding_dim))
            # print('Q_adp_emb,KV_adp_emb: ',Q_adp_emb.size(),KV_adp_emb.size())
            query = torch.cat([query, Q_adp_emb], dim=-1)
            x_contextual = torch.cat([x_contextual, KV_adp_emb], dim=-1)


        # ...
        
        for attn in self.attn_layers:
            # print('\n--- Cross attention layer ---')
            # print('query size before attn:', query.size())
            # print('x_contextual size before attn:', x_contextual.size())
            query = attn(query, dim = dim,x_contextual = x_contextual)
        # print('Extracted feature dim:', query.size())
        return query





class backbone_model(nn.Module):
    def __init__(
        self,
        num_nodes,
        L=12,
        step_ahead=12,
        time_step_per_hour=12,
        C=1,
        out_dim_factor = 1,
        input_embedding_dim=24,
        contextual_input_embedding_dim = None,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        contextual_positions = {},
        horizon_step = 1,
        added_dim_output = 0,
        added_dim_input = 0,
        contextual_kwargs = {},
        Q_num_nodes = 40,
        KV_num_nodes = 13,
        init_adaptive_query_dim = 0 ,
        Early_fusion_names = [],
        Late_fusion_names = [],
        cross_attention = False,
    ):
        super().__init__()
        #  ---  self attributes: ---
        # self.concatenation_late = added_dim_output > 0
        self.num_nodes: int = num_nodes
        self.in_steps = L
        self.out_steps = step_ahead
        self.steps_per_day = 24*time_step_per_hour
        self.input_dim = C
        self.output_dim = out_dim_factor
        self.horizon_step = horizon_step 
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.init_adaptive_query_dim = init_adaptive_query_dim
        self.added_dim_output = added_dim_output
        self.added_dim_input = 0 # added_dim_input
        self.Q_num_nodes = Q_num_nodes 
        self.KV_num_nodes = KV_num_nodes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.Early_fusion_names = Early_fusion_names
        self.Late_fusion_names = Late_fusion_names
        self.cross_attention = cross_attention
    
        self.sum_contextual_dim = 0
        for k in contextual_kwargs.keys():
            concatenation_late_bool = ('attn_kwargs' in contextual_kwargs[k].keys()) and ('concatenation_late' in contextual_kwargs[k]['attn_kwargs'].keys()) and (contextual_kwargs[k]['attn_kwargs']['concatenation_late'])
            # Embedding with concatenation early : 
            if 'emb_dim' in contextual_kwargs[k].keys() and not( concatenation_late_bool ):
                self.sum_contextual_dim = self.sum_contextual_dim + contextual_kwargs[k]['emb_dim']

        Q_input_dim = input_embedding_dim if  self.init_adaptive_query_dim == 0 else self.init_adaptive_query_dim
        KV_input_dim = contextual_input_embedding_dim if contextual_input_embedding_dim is not None else Q_input_dim
        self.Q_model_dim = (
            Q_input_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
            + self.added_dim_input
            + self.sum_contextual_dim
        )
        # print('Q_model_dim:',self.Q_model_dim)
        # print('Q_input_dim:',Q_input_dim)
        # print('tod_embedding_dim:',tod_embedding_dim)
        # print('dow_embedding_dim:',dow_embedding_dim)
        # print('spatial_embedding_dim:',spatial_embedding_dim)
        # print('adaptive_embedding_dim:',adaptive_embedding_dim)
        # print('added_dim_input:',self.added_dim_input)
        # print('sum_contextual_dim:',self.sum_contextual_dim)


        self.KV_model_dim = (
            KV_input_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
            + self.added_dim_input
            + self.sum_contextual_dim
        )



        self.contextual_proj = nn.Linear(self.input_dim, contextual_input_embedding_dim if contextual_input_embedding_dim is not None else input_embedding_dim)

        if self.init_adaptive_query_dim >0:
            self.init_adaptive_query = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.in_steps, self.Q_num_nodes, self.init_adaptive_query_dim)))
            self.input_proj = None
        else:
            self.init_adaptive_query = None
            self.input_proj = nn.Linear(1, input_embedding_dim) 
        #  --- 

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(self.steps_per_day, tod_embedding_dim)
        else:
            self.tod_embedding = None

        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        else:
            self.dow_embedding = None
        
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        else:
            self.node_emb = None


            
        self.contextual_kwargs = contextual_kwargs
        self.contextual_positions = contextual_positions
        self.pos_tod = contextual_positions.get("calendar_timeofday", None)
        self.pos_dow = contextual_positions.get("calendar_dayofweek", None)

        self.contextual_input_embedding =  ContextualInputEmbedding(self.num_nodes,self.contextual_kwargs,
                                                                    self.sum_contextual_dim,self.contextual_positions,
                                                                    self.Early_fusion_names,self.Late_fusion_names)
        
        # --- Temporal Attention Layers :
        if self.cross_attention :
            self.Q_attn_layers_t = nn.ModuleList(
                [
                    SelfAttentionLayer(self.Q_model_dim, feed_forward_dim, num_heads, dropout)
                    for _ in range(num_layers)
                ]
            )
            self.KV_attn_layers_t = nn.ModuleList(
                [
                    SelfAttentionLayer(self.KV_model_dim, feed_forward_dim, num_heads, dropout)
                    for _ in range(num_layers)
                ]
            )
            self.attn_layers_t = None
        else:
            self.attn_layers_t = nn.ModuleList(
                [
                    SelfAttentionLayer(self.Q_model_dim, feed_forward_dim, num_heads, dropout)
                    for _ in range(num_layers)
                ]
            )
            self.Q_attn_layers_t = None
            self.KV_attn_layers_t = None
        # --- 
            
        # --- Spatial  Attention Layers :
        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.Q_model_dim, feed_forward_dim, num_heads, dropout,KV_model_dim = self.KV_model_dim)
                for _ in range(num_layers)
            ]
        )
        # ---

        # --- Adaptive Embedding for added to Query and Key,Value
        if adaptive_embedding_dim > 0:
            if self.cross_attention :
                self.Q_adaptive_embedding = nn.init.xavier_uniform_(
                                nn.Parameter(torch.empty(self.in_steps, self.Q_num_nodes, adaptive_embedding_dim)))
                self.KV_adaptive_embedding = nn.init.xavier_uniform_(
                                nn.Parameter(torch.empty(self.in_steps, self.KV_num_nodes, adaptive_embedding_dim)))
                self.adaptive_embedding = None
            else:
                self.adaptive_embedding = nn.init.xavier_uniform_( 
                                nn.Parameter(torch.empty(self.in_steps, self.num_nodes, adaptive_embedding_dim))
                )
                self.Q_adaptive_embedding = None
                self.KV_adaptive_embedding = None 

        else:
            self.Q_adaptive_embedding = None
            self.KV_adaptive_embedding = None
            self.adaptive_embedding = None
        # ----
        
    def forward(self,  x: Tensor,
                x_contextual: Tensor = None,
                x_calendar : Tensor = None,
                dic_t_emb : Tensor = None,
                dim: int = None
                ) -> Tensor:
        # print('\nStart BACKBONE forward')
        # print('   x shape before input_proj:', x.shape)


        batch_size = x.size(0)

        # ---------- Set Initial Query, Key and Values ---------- : 
        #  --- Adaptive Tensor as Query : 
        if self.init_adaptive_query is not None:
            # print('   generate Query from adaptive tensor')
            query_init = self.init_adaptive_query.expand(size=(batch_size, self.in_steps, self.Q_num_nodes, self.init_adaptive_query_dim))
        
        # --- Use X as query : 
        else:
            # print('   self.input_proj:',self.input_proj)
            if x.dim()==4:
                x = x.permute(0,2,3,1) # [B,C,N,L] -> [B,N,L,C]
            if x.dim()==3:
                x = x.unsqueeze(-1)
            x = self.input_proj(x)
            x = x.transpose(1,2)  
            # print('   x.size() after input proj and transpose: ',x.size())
            query_init  = x

        # --- Use Contextual as Key, Values if exists: 
        if x_contextual is not None:
            # print('   init x_contextual.size(): ',x_contextual.size())
            if x_contextual.dim()==3:
                x_contextual = x_contextual.unsqueeze(-1)
            else:
                x_contextual = x_contextual.transpose(1,3)
            # print('   x_contextual.size() after unsqueeze/transpose: ',x_contextual.size())
            # print('   contextual_proj: ',self.contextual_proj)
            x_contextual = self.contextual_proj(x_contextual)
            # print('   x_contextual.size() after contextual proj and transpose: ',x_contextual.size())
        else:
            x_contextual = query_init
        # ---------- 



        Q_features = torch.empty(query_init.size(0), query_init.size(1), query_init.size(2), 0,dtype=x.dtype, device=x.device) 
        KV_features = torch.empty(x_contextual.size(0), x_contextual.size(1), x_contextual.size(2), 0,dtype=x_contextual.dtype, device=x_contextual.device) 


        # print('\nAdd Embeddings to Query and Key,Value : ')
        # ---------- Calendar Embedding : 
        if (self.tod_embedding is not None) or (self.dow_embedding is not None):
            if x_calendar is None:
                raise ValueError("x_calendar is None. Set args.calendar_types to ['dayofweek', 'timeofday'] and add 'calendar' to dataset_names.")
            # print('    init x_calendar.size(): ',x_calendar.size())
            calendar_features = torch.empty(query_init.size(0), query_init.size(1), 0,dtype=query_init.dtype, device=query_init.device)
            if self.tod_embedding is not None:
                # print('   add TOD embedding')
                tod = x_calendar[..., self.pos_tod]
                tod_emb = self.tod_embedding( (tod * self.steps_per_day).long() )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
                calendar_features = torch.cat([calendar_features, tod_emb], dim=-1)

            if self.dow_embedding is not None:
                # print('   add DOW embedding')
                dow = x_calendar[..., self.pos_dow]
                dow_emb = self.dow_embedding(dow.long())  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
                calendar_features = torch.cat([calendar_features, dow_emb], dim=-1)
            calendar_features = calendar_features.unsqueeze(2)  # [B,L,emb_dim] -> [B,L,1,emb_dim]
            # print('   calendar_features size: ',calendar_features.size())
            Q_features = torch.cat([Q_features, calendar_features.repeat(1,1,Q_features.size(2),1)], dim=-1)
            KV_features = torch.cat([KV_features, calendar_features.repeat(1,1,KV_features.size(2),1)], dim=-1)
        # ---------- 




        # ---------- Adaptive Embedding : 
        #       - If Self Attention 
        if self.adaptive_embedding is not None:
            adp_emb = self.adaptive_embedding.expand(size=(batch_size, self.in_steps, self.num_nodes, self.adaptive_embedding_dim))
            # print('\nadd common adpt embedding')
            # print('   adp_emb size: ',adp_emb.size())
            Q_features = torch.cat([Q_features, adp_emb], dim=-1)
            KV_features = torch.cat([KV_features, adp_emb], dim=-1)

        #       - If Cross Attention 
        if self.Q_adaptive_embedding is not None:
            Q_adp_emb = self.Q_adaptive_embedding.expand(size=(batch_size, self.in_steps, self.Q_num_nodes, self.adaptive_embedding_dim))
            KV_adp_emb = self.KV_adaptive_embedding.expand(size=(batch_size, self.in_steps, self.KV_num_nodes, self.adaptive_embedding_dim))
            # print('\nadd Q_adp_emb and KV_adp_emb embedding')
            # print('   Q_adp_emb size: ',Q_adp_emb.size())
            # print('   KV_adp_emb size: ',KV_adp_emb.size())
            Q_features = torch.cat([Q_features, Q_adp_emb], dim=-1)
            KV_features = torch.cat([KV_features, KV_adp_emb], dim=-1)
        # ----------
            


        # ---------- Other Contextual Embedding : 
        # Add contextual features as inputs (Embedding + concatenation early)
        # DO NOT ADD LATE FEATURE AS IT WILL BE ALREADY ADDED AT THE OUTPUT HEAD
        self.contextual_input_embedding.t_embedding(dic_t_emb)
        early_features = [contextual_i for ds_name, contextual_i in self.contextual_input_embedding.projected_contextual.items() if ds_name in self.Early_fusion_names]
        early_features = torch.cat(early_features, dim=-1) if len(early_features)>0 else torch.empty(0)
        late_features = [contextual_i for ds_name, contextual_i in self.contextual_input_embedding.projected_contextual.items() if ds_name in self.Late_fusion_names]
        late_features = torch.cat(late_features, dim=-1) if len(late_features)>0 else torch.empty(0)
        
        if late_features.size(-1) >0:
            raise NotImplementedError("Late fusion of contextual features should be forbiden here as all the backbone model for heterogenous spatial unit will produce repeated embedding of contextual data which will be concatenated at the output head of the main traffic model.")
        if early_features.size(-1) >0:
            # print('\nadd contextual early features')
            Q_features = self.contextual_input_embedding.concat_features(Q_features, repeat_transpose(x = early_features, n_repeat = self.Q_num_nodes))
            KV_features = self.contextual_input_embedding.concat_features(KV_features, repeat_transpose(x = early_features, n_repeat = self.KV_num_nodes))
            # print('   Q_features.size() after adding contextual early features : ',Q_features.size())
            # print('   KV_features.size() after adding contextual early features : ',KV_features.size())
        # ----------


        query_init = torch.cat([query_init, Q_features], dim=-1)
        x_contextual = torch.cat([x_contextual, KV_features], dim=-1)

        # FEATURE EXTRACTION
        # print('\nStart Feature Extraction with Attention layers')
        # print('   query_init.size() before attn-layers: ',query_init.size())
        # print('   x_contextual.size() before attn-layers: ',x_contextual.size())
        # print('   cross_attention layer:', self.Q_attn_layers_t[0])
        # ----------  Temporal & Spatial Attention Layers  : 

        #if Cross Attention,  Need to first project into spatial dimension  ==> INVERSE ORDER OF ATTENTION LAYERS
        if self.cross_attention:
            # print('   Use Cross Attention layers')
            for attn in self.Q_attn_layers_t:
                # print(    'query_init size before attn:', query_init.size())
                # print(    'attn_layer:', attn)
                query_init = attn(query_init, dim=1)
            for attn in self.KV_attn_layers_t:
                # print(    '\nx_contextual size before attn:', x_contextual.size())
                # print(    'attn_layer:', attn)
                x_contextual = attn(x_contextual, dim=1) 

            for attn in self.attn_layers_s:
                # print(    '\nquery_init size before attn:', query_init.size())
                # print(    'x_contextual size before attn:', x_contextual.size())
                # print(    'attn_layer:', attn)
                query_init = attn(query_init, dim=2, x_contextual = x_contextual)


        # Else, keep normal STAEformer:  
        else:
            # print('   Use Self Attention layers')
            for attn in self.attn_layers_t:
                query_init = attn(query_init, dim=1)
            for attn in self.attn_layers_s:
                query_init = attn(query_init, dim=2)

        # (batch_size, in_steps, num_nodes, model_dim)
        # print('   x.size() after T-attn and S-attn : ',query_init.size())
        # ---------- 

        # print('input embedding dim:', self.input_embedding_dim)
        # print('tod_embedding_dim:', self.tod_embedding_dim)
        # print('dow_embedding_dim:', self.dow_embedding_dim)
        # print('spatial_embedding_dim:', self.spatial_embedding_dim)
        # print('adaptive_embedding_dim:', self.adaptive_embedding_dim)
        # print('added_dim_input:', self.added_dim_input)
        # print('added_dim_output:', self.added_dim_output)
        # print('sum_contextual_dim:', self.sum_contextual_dim)
        # print('Total model_dim:', self.Q_model_dim)
        # print('input_embedding_dim:', self.input_embedding_dim ) 
        # print('init_adaptive_query_dim: ', self.init_adaptive_query_dim)
        return query_init




if __name__ == "__main__":
    import torch
    # Debug JIT scripting per module
    print("Testing AttentionLayer...")
    try:
        attn = AttentionLayer(model_dim=32, num_heads=4)
        torch.jit.script(attn)
        print("→ AttentionLayer scripted OK")
    except Exception as e:
        print("→ AttentionLayer JIT error:", e)

    print("Testing SelfAttentionLayer...")
    try:
        self_attn = SelfAttentionLayer(model_dim=32, feed_forward_dim=128, num_heads=4, dropout=0.1)
        torch.jit.script(self_attn)
        print("→ SelfAttentionLayer scripted OK")
    except Exception as e:
        print("→ SelfAttentionLayer JIT error:", e)

    print("Testing STAEformer...")
    try:
        model = STAEformer(num_nodes=10, L=12, step_ahead=12, time_step_per_hour=12, C=1, out_dim_factor=1)
        dummy_x = torch.randn(1,1,10,12)
        dummy_cal = torch.zeros(1,12,2)
        _ = model(dummy_x,x_calendar= dummy_cal)  # eager forward
        torch.jit.script(model)
        print("→ STAEformer scripted OK")
    except Exception as e:
        print("→ STAEformer JIT error:", e)

