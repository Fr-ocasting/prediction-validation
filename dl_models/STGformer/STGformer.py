import torch.nn as nn
import torch
from torch import Tensor
import numpy as np
from timm.models.vision_transformer import Mlp
import torch.nn.functional as F
from typing import Optional,List

class FastAttentionLayer(nn.Module):
    def __init__(self, model_dim: int, num_heads: int=8, qkv_bias: bool=False, kernel: int=1):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        assert model_dim % num_heads == 0, "final embedding dim must be divisible by num_heads"
        self.head_dim = model_dim // num_heads  

        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)

        self.out_proj = nn.Linear(
            2 * model_dim if kernel != 12 else model_dim, model_dim
        )
        # self.proj_in = nn.Conv2d(model_dim, model_dim, (1, kernel), 1, 0) if kernel > 1 else nn.Identity()
        self.fast = 1

    def forward(self, x:  torch.Tensor, edge_index: Optional[torch.Tensor] = None, dim: int = 0) -> torch.Tensor:
        
        # x = self.proj_in(x.transpose(1, 3)).transpose(1, 3)
        query, key, value = self.qkv(x).chunk(3, -1)
        qs = torch.stack(torch.split(query, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=dim, end_dim=dim + 1
        )
        ks = torch.stack(torch.split(key, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=dim, end_dim=dim + 1
        )
        vs = torch.stack(torch.split(value, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=dim, end_dim=dim + 1
        )
        if self.fast:
            out_s = self.fast_attention(x, qs, ks, vs, dim=dim)
        else:
            out_s = self.normal_attention(x, qs, ks, vs, dim=dim)
        if x.size(1) > 1:
            qs = torch.stack(
                torch.split(query.transpose(1, 2), self.head_dim, dim=-1), dim=-2
            ).flatten(start_dim=dim, end_dim=dim + 1)
            ks = torch.stack(
                torch.split(key.transpose(1, 2), self.head_dim, dim=-1), dim=-2
            ).flatten(start_dim=dim, end_dim=dim + 1)
            vs = torch.stack(
                torch.split(value.transpose(1, 2), self.head_dim, dim=-1), dim=-2
            ).flatten(start_dim=dim, end_dim=dim + 1)
            if self.fast:
                out_t = self.fast_attention(
                    x.transpose(1, 2), qs, ks, vs, dim=dim
                ).transpose(1, 2)
            else:
                out_t = self.normal_attention(
                    x.transpose(1, 2), qs, ks, vs, dim=dim
                ).transpose(1, 2)
            out = torch.cat([out_s, out_t], -1)
            out = self.out_proj(out)
        else:
            out = self.out_proj(out_s)

        return out

    def fast_attention(self, x: torch.Tensor, qs: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor, dim: int = 0) -> torch.Tensor:
        qs = nn.functional.normalize(qs, dim=-1)
        ks = nn.functional.normalize(ks, dim=-1)
        N = qs.shape[1]
        b, l = x.shape[dim : dim + 2]

        # numerator
        kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)
        attention_num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs)  # [N, H, D]
        attention_num = attention_num+N * vs

        # denominator
        all_ones = torch.ones([ks.shape[1]]).to(ks.device)
        ks_sum = torch.einsum("blhm,l->bhm", ks, all_ones)
        attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape)
        )  # [N, H, 1]
        attention_normalizer = attention_normalizer + torch.ones_like(attention_normalizer) * N
        out = attention_num / attention_normalizer  # [N, H, D]
        out = torch.unflatten(out, dim, (b, l)).flatten(start_dim=3)
        return out

    def normal_attention(self, x: torch.Tensor, qs: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor, dim: int = 0) -> torch.Tensor:
        b, l = x.shape[dim : dim + 2]
        qs, ks, vs = qs.transpose(1, 2), ks.transpose(1, 2), vs.transpose(1, 2)
        x = (
            torch.nn.functional.scaled_dot_product_attention(qs, ks, vs)
            .transpose(-3, -2)
            .flatten(start_dim=-2)
        )
        x = torch.unflatten(x, dim, (b, l)).flatten(start_dim=3)
        return x

"""
class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, qkv_bias=False, kernel=1):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads

        self.head_dim = model_dim // num_heads

        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        query, key, value = self.qkv(x).chunk(3, -1)
        qs = torch.stack(torch.split(query, self.head_dim, dim=-1), dim=-3)
        ks = torch.stack(torch.split(key, self.head_dim, dim=-1), dim=-3)
        vs = torch.stack(torch.split(value, self.head_dim, dim=-1), dim=-3)
        x = (
            torch.nn.functional.scaled_dot_product_attention(qs, ks, vs)
            .transpose(-3, -2)
            .flatten(start_dim=-2)
        )
        x = self.out_proj(x)
        return x
"""



class GraphPropagate(nn.Module):
    def __init__(self, Ks, gso, dropout = 0.2):
        super(GraphPropagate, self).__init__()
        self.Ks = Ks
        self.gso = gso
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, graph):
        if self.Ks < 1:
            raise ValueError(
                f"ERROR: Ks must be a positive integer, received {self.Ks}."
            )
        x_k = x; x_list = [x]
        for k in range(1, self.Ks):
            x_k = torch.einsum("thi,btij->bthj", graph, x_k.clone())
            x_list.append(self.dropout(x_k))

        return x_list


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        mlp_ratio: float=2.0,
        num_heads: int =8,
        dropout: float=0,
        mask: bool=False,
        kernel: int=3,
        supports: Optional[List[torch.Tensor]]=None,
        order: int=2,
    ):
        super().__init__()
        self.locals = GraphPropagate(Ks=order, gso=supports)
        """
        self.attn = nn.ModuleList(
            [
                FastAttentionLayer(model_dim, num_heads, mask, kernel=kernel)
                for _ in range(order)
            ]
        )
        self.pws = nn.ModuleList(
            [nn.Linear(model_dim, model_dim) for _ in range(order)]
        )

        for i in range(0, order):
            nn.init.constant_(self.pws[i].weight, 0)
            nn.init.constant_(self.pws[i].bias, 0)
        """

        ## NEW FOR JIT
        self.layers = nn.ModuleDict({
            str(i): nn.ModuleDict({
                "att": FastAttentionLayer(model_dim, num_heads, mask, kernel=kernel),
                "pw":  nn.Linear(model_dim, model_dim)
            })
            for i in range(order)
        })

        for i in range(0, order):
            nn.init.constant_(self.layers[str(i)]['pw'].weight, 0)
            nn.init.constant_(self.layers[str(i)]['pw'].bias, 0)
        ## NEW FOR JIT



        self.kernel = kernel
        self.fc = Mlp(
            in_features=model_dim,
            hidden_features=int(model_dim * mlp_ratio),
            act_layer=nn.ReLU,
            drop=dropout,
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale: List[float] = [1.0, 0.01, 0.001]
        self.order: int = order

    def forward(self, x, graph):
        x_loc = self.locals(x, graph)
        c = x_glo = x

        ## NEW FOR JIT
        for k, layer in self.layers.items():
            z = x_loc[int(k)]
            att_outputs = layer["att"](z)
            pw  = layer["pw"](c)
            sc  = self.scale[int(k)]
            x_glo = x_glo + att_outputs * pw * sc
            c = att_outputs

        ## NEW FOR JIT
        """
        for attn_module, pw_module, scale_value, z in zip(self.attn, self.pws, self.scale, x_loc):
            att_outputs = attn_module(z)
            x_glo = x_glo + att_outputs * pw_module(c) * scale_value
            c = att_outputs
        """

        x = self.ln1(x + self.dropout(x_glo))
        x = self.ln2(x + self.dropout(self.fc(x)))
        return x


class STGformer(nn.Module):
    def __init__(
        self,
        n_vertex,
        L=12,
        step_ahead=12,
        time_step_per_hour=12,
        C=1,
        out_dim_factor = 1,
        input_embedding_dim=24,
        tod_embedding_dim=12,
        dow_embedding_dim=12,
        adaptive_embedding_dim=12,
        num_heads=4,
        supports=None,
        num_layers=3,
        dropout=0.1,
        mlp_ratio=2,
        dropout_a=0.3,
        kernel_size=[1],
        contextual_positions = {}
    ):
        super().__init__()

        self.num_nodes: int = n_vertex
        self.in_steps: int = L
        self.out_steps: int = step_ahead
        self.steps_per_day: int = 24*time_step_per_hour
        self.input_dim: int = C
        self.output_dim: int = out_dim_factor
        self.input_embedding_dim: int = input_embedding_dim
        self.tod_embedding_dim: int = tod_embedding_dim
        self.dow_embedding_dim: int = dow_embedding_dim
        self.adaptive_embedding_dim: int = adaptive_embedding_dim
        self.model_dim: int = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads: int = num_heads
        self.num_layers: int = num_layers

        self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)
        if self.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        else:
            self.tod_embedding = None
            
        if self.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        else:
            self.dow_embedding = None

        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.in_steps, self.num_nodes, self.adaptive_embedding_dim))
            )
        else:
            self.adaptive_embedding = None

        self.dropout = nn.Dropout(dropout_a)
        # self.dropout = DropPath(dropout_a)
        self.pooling = nn.AvgPool2d(kernel_size=(1, kernel_size[0]), stride=1)
        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim,
                    mlp_ratio,
                    self.num_heads,
                    dropout,
                    kernel=size,
                    supports=supports,
                )
                for size in kernel_size
            ]
        )

        """ # Modification:
        self.encoder_proj = nn.Linear(
            (self.in_steps - sum(k - 1 for k in kernel_size)) * self.model_dim,
            self.model_dim,
        )
        """
        self.encoder_proj = nn.Linear(
            (self.in_steps - (kernel_size[0] - 1)) * self.model_dim,
            self.model_dim,
        )


        self.kernel_size = kernel_size[0]

        self.encoder = nn.ModuleList(
            [
                Mlp(
                    in_features=self.model_dim,
                    hidden_features=int(self.model_dim * mlp_ratio),
                    act_layer=nn.ReLU,
                    drop=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(self.model_dim, self.out_steps * self.output_dim)
        # self.temporal_proj = TCNLayer(self.model_dim, self.model_dim, max_seq_length=in_steps)
        self.temporal_proj = nn.Conv2d(
            self.model_dim, self.model_dim, (1, kernel_size[0]), 1, 0
        )
        self.pos_tod = contextual_positions.get("calendar_timeofday", None)
        self.pos_dow = contextual_positions.get("calendar_dayofweek", None)

    def forward(self, x: Tensor,
                x_vision: Optional[Tensor] = None, 
                x_calendar: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: (batch_size, input_dim, num_nodes, in_steps)  
               -> Need to permute x cause expected size is: (batch_size, in_steps, num_nodes, input_dim) 
            x_vision: None
            x_calendar: (batch_size, in_steps, 2) last dim = 2 cause channels are[dayofweek,timeofday]

        x_calendar.size() has dim has to correspond to x.size(), then we repeat it
        """
        if x_vision is not None:
            raise NotImplementedError("tackling x_vision has not been implemented")
        
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
            
            #print('x after projection: ',x.size(), 'expected (batch_size, in_steps, num_nodes, input_embedding_dim)')

            # Init features with empty tensor with 0 channels
            features = torch.empty(x.size(0), x.size(1), x.size(2), 0,dtype=x.dtype, device=x.device)  

            if self.tod_embedding is not None:
                tod = x_calendar[..., self.pos_tod]
                tod_emb = self.tod_embedding( (tod * self.steps_per_day).long() )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
                features = torch.cat([features, tod_emb], -1)

            if self.dow_embedding is not None:
                dow = x_calendar[..., self.pos_dow]
                dow_emb = self.dow_embedding(dow.long())  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
                features = torch.cat([features, dow_emb], -1)

            if self.adaptive_embedding is not None:
                adp_emb = self.adaptive_embedding.expand(
                    size=(batch_size, self.in_steps, self.num_nodes, self.adaptive_embedding_dim)
                )
                features = torch.cat([features, self.dropout(adp_emb)], -1)

            x = torch.cat([x] + [features], dim=-1)
                
            # (batch_size, in_steps, num_nodes, model_dim)
            #print('x after concatenation with embedded features: ',x.size(), 'expected (batch_size, in_steps, num_nodes, model_dim)')
            #print('x.transpose: ',x.transpose(1, 3).size())
            #print('temporal proj: ',self.temporal_proj)
            x = self.temporal_proj(x.transpose(1, 3)).transpose(1, 3)
            graph = torch.matmul(self.adaptive_embedding, self.adaptive_embedding.transpose(1, 2))
            graph = self.pooling(graph.transpose(0, 2)).transpose(0, 2)
            graph = F.softmax(F.relu(graph), dim=-1)
            for attn in self.attn_layers_s:
                x = attn(x, graph)
            x = self.encoder_proj(x.transpose(1, 2).flatten(-2))
            for layer in self.encoder:
                x = x + layer(x)
            # (batch_size, in_steps, num_nodes, model_dim)
            #out = self.output_proj(x).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
            out = self.output_proj(x).contiguous().reshape(-1, self.num_nodes, self.out_steps, self.output_dim)
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
            out = out.permute(0,3,2,1)  #  (batch_size, output_dim, num_nodes, out_steps )

            return out