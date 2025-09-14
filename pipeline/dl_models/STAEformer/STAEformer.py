"""
STAEformer from  https://github.com/xdzhelheim/staeformer
"""
from torch import Tensor
import torch.nn as nn
import torch
from typing import Optional,List



class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

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
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
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
    def __init__(
        self,
        num_nodes,
        contextual_kwargs = {},
        sum_contextual_dim = 0,
        contextual_positions = {},
    ):

        super().__init__()
        self.num_nodes: int = num_nodes
        self.sum_contextual_dim = sum_contextual_dim
        self.contextual_kwargs = contextual_kwargs
        self.contextual_positions = contextual_positions
        self.contextual_emb = nn.ModuleDict()
        self.contextual_spatial_proj = nn.ModuleDict()

        if self.sum_contextual_dim>0:
            for ds_name, kwargs_i in self.contextual_kwargs.items():
                if 'emb_dim' in kwargs_i.keys():
                    self.contextual_emb[ds_name] = nn.Linear(kwargs_i['C'], kwargs_i['emb_dim'])
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


    def forward(self, 
                features : Tensor, 
                contextual: Optional[list[Tensor]]= None,
                ) -> Tensor:


        # Heterogeneous or Homogenous spatial units others contextual features: 
        for ds_name, _ in self.contextual_emb.items():
            # print('\nds_name: ',ds_name)
            contextual_i = contextual[self.contextual_positions[ds_name]] 
            # print(f'contextual shape before embedding:', contextual_i.size())
            # Align the dimensions
            if contextual_i.dim() ==3:
                contextual_i = contextual_i.unsqueeze(-1) # [B,P,L] -> [B,P,L,1]
            # Embedding on channel dim: 
            contextual_i = self.contextual_emb[ds_name](contextual_i)  # [B,P,L,C] -> [B,P,L,emb_dim]

            # --- Spatial Projection : 
            # Repeat Tensor if common for all nodes : 
            if contextual_i.size(1) == 1:
                contextual_i = contextual_i.repeat(1,self.num_nodes,1,1) # [B,1,L,emb_dim] -> [B,N,L,emb_dim]
                contextual_i = contextual_i.transpose(1,2) # [B,N,L,emb_dim] -> [B,L,N,emb_dim]

            # Otherwise spatial projection if n_spatial_unit != num_nodes
            if ds_name in self.contextual_spatial_proj.keys():
                contextual_i = self.contextual_spatial_proj[ds_name](contextual_i.permute(0,3,2,1))  # [B,P,L,emb_dim] -permute-> [B,emb_dim,L,P] -> [B,emb_dim,L,N]
                contextual_i = contextual_i.permute(0,2,3,1)  # [B,emb_dim,L,N] -> [B,L,N,emb_dim]
            else:
                contextual_i = contextual_i.transpose(1,2)
            features = torch.cat([features, contextual_i], dim=-1)
        return features


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
    ):
        super().__init__()
        #  ---  self attributes: ---
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
        self.sum_contextual_dim = sum(contextual_kwargs[k]['emb_dim'] for k in contextual_kwargs.keys() if 'emb_dim' in contextual_kwargs[k].keys())
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
            + added_dim_input
            + self.sum_contextual_dim
        )
        # print('input embedding dim:', self.input_embedding_dim)
        # print('tod_embedding_dim:', self.tod_embedding_dim)
        # print('dow_embedding_dim:', self.dow_embedding_dim)
        # print('spatial_embedding_dim:', self.spatial_embedding_dim)
        # print('adaptive_embedding_dim:', self.adaptive_embedding_dim)
        # print('added_dim_input:', self.added_dim_input)
        # print('sum_contextual_dim:', self.sum_contextual_dim)
        self.output_model_dim = self.model_dim + self.added_dim_output
        
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
                                                                    self.sum_contextual_dim,self.contextual_positions)

        # self.contextual_kwargs = contextual_kwargs
        # self.contextual_emb = nn.ModuleDict()
        # self.contextual_spatial_proj = nn.ModuleDict()
        # if self.sum_contextual_dim>0:
        #     for ds_name, kwargs_i in self.contextual_kwargs.items():
        #         if 'emb_dim' in kwargs_i.keys():
        #             self.contextual_emb[ds_name] = nn.Linear(kwargs_i['C'], kwargs_i['emb_dim'])
        #             if ('n_spatial_unit' in kwargs_i.keys()) and (kwargs_i['n_spatial_unit']) is not None and ((not 'repeat_spatial' in kwargs_i.keys()) or (not kwargs_i['repeat_spatial'])):
        #                 self.contextual_spatial_proj[ds_name] = nn.Linear(kwargs_i['n_spatial_unit'], self.num_nodes)


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


# ---- PREDICTION SUBWAY-OUT ----

# --- NATIVEMENT STAEformer : 
# subway-out --Embedding--> Z_subway_out
# calendar   --Embedding--> Z_calendar
# input_STAEformer = concat(Z_subway_out, Z_calendar)


# --- MODIFIED STAEformer : 
# subway-out (1 channel) --Embedding--> Z_subway_out                                                Size:     [B, Z_sub_out , N, L])
# subway-in (1 channel)  --Embedding--> Z_subway_in                                                 Size:     [B, Z_sub_in  , N, L]) ---- ## ADDED  ##-----
# bike-dropoff(1 channel) --Embedding--> Z_bike_dropoff                                             Size:     [B, Z_bike_out, P, L]) ---- ## ADDED  ##-----
# 

#     Embedding spatial (display all possibilities):
#     - With spatial Embedding      
#         [B, Z_bike_out, P, L] --Fully Connected Layer(P,N) --> [B, Z_bike_out, N, L]   
#     - With spatial attention : 
#         Query (Subway-out), Key (bike-out), Value (bike-out)  -- Attetion On Spatial Dim --> [B, Z_bike_out, N, L]

# calendar  (2 channel)  --Embedding--> Z_calendar     ( 2*6 channels  [B, Z_calendar, 1, L]) --Repeat x N--> [B, Z_calendar, N, L]

# - Block concat on Channel dim :
# input_STAEformer = concat(Z_subway_out, Z_subway_in, Z_calendar, Z_bike_dropoff)                  Size:     [B,Z_tot,N,L]

# - Feed embeding: 
# input_STAEformer --Feed--> STAEformer --> output_STAEformer

# -----


# --- BEFORE STAEformer --- : 
# concat[subway-out,subway_in]) --Embedding--> Z_subway_out_in
# calendar   --Embedding--> Z_calendar
# input_STAEformer = concat(Z_subway_out, Z_subway_in, Z_calendar)



#  [B,1,N,L] , [B,1,N,L]  --STACK CHANNEL--> [B,2,N,L]   --Channel Proj  --> [B,C,N,L]
#  [B,1,N,L] , [B,1,N,L]  --Concat--> [B,1,N,2L] 





        
    def forward(self,  x: Tensor,
                x_vision: Optional[Tensor] = None, 
                x_calendar: Optional[Tensor] = None,
                contextual: Optional[list[Tensor]]= None,
                ) -> Tensor:
        
        # print('x shape before input_proj:', x.shape)

        if x_calendar is None:
            raise ValueError("x_calendar is None. Set args.calendar_types to ['dayofweek', 'timeofday'] and add 'calendar' to dataset_names.")

        else: 
            x = x.permute(0,3,2,1) # [B,C,N,L] -> [B,L,N,C]
            if x_calendar.size(-1) != 2:
                raise ValueError(f"Expected x_calendar.size(-1) == 2, but got {x_calendar.size(-1)}. Set args.calendar_types to ['dayofweek', 'timeofday'] and add 'calendar' to dataset_names.")
            if x_calendar.dim() ==3:
                x_calendar = x_calendar.unsqueeze(2)  # [B,L,2] -> [B,L,1,2]
            x_calendar = x_calendar.repeat(1,1,self.num_nodes,1) # [B,L,1,2]-> [B,L,N,2]

            # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
            batch_size = x.shape[0]
            x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
            # print('x size after input_proj:', x.size())
            #print('x_calendar shape:', x_calendar.shape)
            
            # Init features with empty tensor with 0 channels
            features = torch.empty(x.size(0), x.size(1), x.size(2), 0,dtype=x.dtype, device=x.device)  

            #print('features shape:', features.shape)
            if self.tod_embedding is not None:
                tod = x_calendar[..., self.pos_tod]
                tod_emb = self.tod_embedding( (tod * self.steps_per_day).long() )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
                features = torch.cat([features, tod_emb], dim=-1)

            if self.dow_embedding is not None:
                dow = x_calendar[..., self.pos_dow]
                dow_emb = self.dow_embedding(dow.long())  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
                features = torch.cat([features, dow_emb], dim=-1)

            if self.node_emb is not None:
                spatial_emb = self.node_emb.expand(batch_size, self.in_steps, self.num_nodes, self.spatial_embedding_dim)
                features = torch.cat([features, spatial_emb],dim= -1)

            if self.adaptive_embedding is not None:
                adp_emb = self.adaptive_embedding.expand(size=(batch_size, self.in_steps, self.num_nodes, self.adaptive_embedding_dim))
                features = torch.cat([features, adp_emb], dim=-1)
            
            # Add contextual features as inputs (Embedding + concatenation early)
            # print('features size before contextual: ',features.size())
            features = self.contextual_input_embedding(features,contextual)
            # print('features size after contextual: ',features.size())


            if x_vision is not None:
                # [B,N,L,C] ->   [B,L,N,C] 
                features = torch.cat([features, x_vision], dim=-1)
            # print('features size after vision concat: ',features.size())



            x = torch.cat([x, features], dim=-1)
            # print('x size after adding all embeddings:', x.size())


            for attn in self.attn_layers_t:
                # print('\n--- Temporal attention layer ---')
                x = attn(x, dim=1)
            for attn in self.attn_layers_s:
                # print('\n--- Spatial attention layer ---')
                x = attn(x, dim=2)
            # (batch_size, in_steps, num_nodes, model_dim)
            # print('x.size(): ',x.size())

            if x_vision is not None and x_vision.dim()==3: 
                raise NotImplementedError("x_vision with 3 dims  and with concatenation late. Need to finish the implementation")
                x_vision = x_vision.unsqueeze().permute(0,3,2,1)
                x = torch.cat([x, x_vision], dim=-1)
                # print("x.size() after concat 'vision': ",x.size())


            if self.temporal_proj is None:
                out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
                out = out.reshape( batch_size, self.num_nodes, self.in_steps * self.model_dim)
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
        self, model_dim, feed_forward_dim=2048, num_heads=8,num_layers = 3, 
        dropout=0, 
        mask=False,
        steps_per_day = 288,
        tod_embedding_dim=0,
        dow_embedding_dim=0,
        pos_tod = None,
        pos_dow = None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, model_dim) 
        self.contextual_proj = nn.Linear(1, model_dim)
        self.attn_layers = nn.ModuleList(
            [
                SelfAttentionLayer(model_dim+tod_embedding_dim+dow_embedding_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.steps_per_day = steps_per_day

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



    def forward(self, x: Tensor, x_contextual: Tensor = None,x_calendar : Tensor = None,dim: int = -2,) -> Tensor:
        if x.dim()==3:
            x = x.unsqueeze(-1)
            x = self.input_proj(x)
            x  = x.transpose(1,2)

        if x_contextual is not None and x_contextual.dim()==3:
            x_contextual = x_contextual.unsqueeze(-1)
            x_contextual = self.contextual_proj(x_contextual)
            x_contextual  = x_contextual.transpose(1,2) if x_contextual is not None else None
        

        calendar_features = torch.empty(x.size(0), x.size(1),1, 0,dtype=x.dtype, device=x.device)
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

            x = torch.cat([x, calendar_features.repeat(1,1,x.size(2),1)], dim=-1)
            x_contextual = torch.cat([x_contextual, calendar_features.repeat(1,1,x_contextual.size(2),1)], dim=-1)

        
        for attn in self.attn_layers:
            x = attn(x, dim = dim,x_contextual = x_contextual)
        return x

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

