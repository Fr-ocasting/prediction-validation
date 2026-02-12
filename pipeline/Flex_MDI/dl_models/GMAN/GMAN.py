import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple
from torch import Tensor

import sys 
import os 
import importlib 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..','..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...
from pipeline.Flex_MDI.dl_models.STAEformer.STAEformer import ContextualInputEmbedding,repeat_transpose


class conv2d_(nn.Module): 
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class STEmbedding(nn.Module):
    '''
    spatio-temporal embedding
    SE:     [num_vertex, D]
    TE:     [batch_size, num_his + num_pred, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, num_his + num_pred, num_vertex, D]
    '''

    def __init__(self, D, bn_decay, steps_per_day=288): # ##### MODIFIE: steps_per_day param added
        super(STEmbedding, self).__init__()
        self.steps_per_day = steps_per_day
        self.FC_se = FC(
            input_dims=[D, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)

        self.FC_te = FC(
            input_dims=[steps_per_day+7, D], units=[D, D], activations=[F.relu, None], # ##### MODIFIED: input_dims dynamic
            bn_decay=bn_decay) 

    def forward(self, SE, TE):
        # spatial embedding
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.FC_se(SE)
        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7, device=TE.device) 
        timeofday = torch.empty(TE.shape[0], TE.shape[1], self.steps_per_day, device=TE.device)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % self.steps_per_day, self.steps_per_day) 
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(dim=2)
        # print('OHE TE[0]: ',TE[0])
        TE = self.FC_te(TE)
        # print('TE[0]: ',TE[0])
        del dayofweek, timeofday
        return SE + TE


class spatialAttention(nn.Module):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''
    def __init__(self, K, d, bn_decay):
        super(spatialAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, STE):
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # [K * batch_size, num_step, num_vertex, num_vertex]
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1) # orginal K, change to batch_size
        X = self.FC(X)
        del query, key, value, attention
        return X

class temporalAttention(nn.Module):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''
    
    def __init__(self, K, d, bn_decay, mask=True):
        super(temporalAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.mask = mask
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, STE):
        batch_size_ = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_step, d]
        # key:   [K * batch_size, num_vertex, d, num_step]
        # value: [K * batch_size, num_vertex, num_step, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_step, num_step]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        # mask attention score
        if self.mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            mask = torch.ones(num_step, num_step, device=X.device) 
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, torch.tensor(-2 ** 15 + 1,
             dtype=attention.dtype, 
             device=attention.device)) # ##### MODIFIE: device and tensor
        # softmax
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size_, dim=0), dim=-1) # orginal K, change to batch_size
        X = self.FC(X)
        return X


class gatedFusion(nn.Module):
    '''
    gated fusion
    HS:     [batch_size, num_step, num_vertex, D]
    HT:     [batch_size, num_step, num_vertex, D]
    D:      output dims
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, D, bn_decay):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        del XS, XT, z
        return H


class STAttBlock(nn.Module):
    def __init__(self, K, d, bn_decay, mask=False):
        super(STAttBlock, self).__init__()
        self.spatialAttention = spatialAttention(K, d, bn_decay)
        self.temporalAttention = temporalAttention(K, d, bn_decay, mask=mask)
        self.gatedFusion = gatedFusion(K * d, bn_decay)

    def forward(self, X, STE):
        HS = self.spatialAttention(X, STE)
        HT = self.temporalAttention(X, STE)
        H = self.gatedFusion(HS, HT)
        del HS, HT
        return torch.add(X, H)


class transformAttention(nn.Module):    
    '''
    transform attention mechanism
    X:        [batch_size, num_his, num_vertex, D]
    STE_his:  [batch_size, num_his, num_vertex, D]
    STE_pred: [batch_size, num_pred, num_vertex, D]
    K:        number of attention heads
    d:        dimension of each attention outputs
    return:   [batch_size, num_pred, num_vertex, D]
    '''

    def __init__(self, K, d, bn_decay):
        super(transformAttention, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, STE_his, STE_pred):
        batch_size = X.shape[0]
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(STE_pred)
        key = self.FC_k(STE_his)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_pred, d]
        # key:   [K * batch_size, num_vertex, d, num_his]
        # value: [K * batch_size, num_vertex, num_his, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_pred, num_his]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_pred, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class GMAN(nn.Module):
    '''
    GMAN
        X :                    [batch_size, num_his, num_vertx]
        TE :                   [batch_size, num_his + num_pred, 2] (time-of-day, day-of-week)
        SE :                   [num_vertex, K * d]
        num_his :              number of history steps
        num_pred :             number of prediction steps
        steps_per_day :        one day is divided into 24*time_step_per_hour steps
        nb_STAttblocks :       number of STAtt blocks in the encoder/decoder
        K :                    number of attention heads
        d :                    dimension of each attention head outputs
        return :               [batch_size, num_pred, num_vertex]
    '''

    def __init__(self, args, SE: Optional[Tensor] = None):
        super(GMAN, self).__init__()
        # ##### MODIFIE: Parsing from args #####
        self.num_nodes = args.num_nodes
        self.L = args.L  # input sequence length (num_his)
        self.step_ahead = args.step_ahead # output sequence length (num_pred)
        self.input_dim = args.C # input channels
        self.horizon_step = args.horizon_step
        
        bn_decay = args.bn_decay 
        self.steps_per_day = 24*args.time_step_per_hour
        
        nb_STAttblocks = args.nb_STAttblocks 
        K = args.num_heads 
        d = args.head_dim 
        D = K * d
        
        self.num_his = self.L
        
        # ##### MODIFIE: Contextual Embedding Init #####
        self.contextual_kwargs = args.contextual_kwargs
        self.contextual_positions = args.contextual_positions
        self.pos_tod = self.contextual_positions.get("calendar_timeofday", None)
        self.pos_dow = self.contextual_positions.get("calendar_dayofweek", None)
        self.contextual_input_embedding = ContextualInputEmbedding(
            self.num_nodes,
            self.contextual_kwargs,
            self.contextual_positions,
            args.Early_fusion_names,
            args.Late_fusion_names
        )
        
        # Calculate added dims
        early_dim = 0
        late_dim = 0
        for ds_name, kwargs_i in self.contextual_kwargs.items():
            if 'emb_dim' in kwargs_i:
                 # Check fusion list in args or contextual_input_embedding
                if ds_name in args.Early_fusion_names:
                    early_dim += kwargs_i['emb_dim']
                elif ds_name in args.Late_fusion_names:
                    late_dim += kwargs_i['emb_dim']


        self.SE = nn.Parameter(SE, requires_grad=True) # Make it parameter

        self.STEmbedding = STEmbedding(D, bn_decay, steps_per_day=self.steps_per_day)
        
        self.STAttBlock_1 = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(nb_STAttblocks)])
        self.STAttBlock_2 = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(nb_STAttblocks)])
        self.transformAttention = transformAttention(K, d, bn_decay)
        
        # ##### MODIFIED: FC_1 input dims adapted for C + Early Fusion #####
        self.FC_1 = FC(input_dims=[self.input_dim + early_dim, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        
        # ##### MODIFIED: FC_2 input dims adapted for D + Late Fusion #####
        self.FC_2 = FC(input_dims=[D + late_dim, D], units=[D, 1], activations=[F.relu, None],
                       bn_decay=bn_decay)

    def forward(self, x: Tensor,
                x_vision_early: Optional[Tensor] = None,
                x_vision_late: Optional[Tensor] = None,
                x_calendar: Optional[Tensor] = None,
                contextual: Optional[list[Tensor]]= None) -> Tensor:
        
        # ##### MODIFIED: 
        # Input Reshape [B, C, N, L] -> [B, L, N, C]
        if x.dim() == 4:
             x = x.permute(0, 3, 2, 1) # -> [B, L, N, C]
        
        batch_size = x.shape[0]
        self.contextual_input_embedding(contextual)

        # Early fusion : 
        x = self.contextual_input_embedding.concat_features(x, self.contextual_input_embedding.early_features)
        if x_vision_early is not None:
             # Assuming x_vision_early matches dimensions or needs permutation
             if x_vision_early.dim() == 4 and x_vision_early.size(1) != x.size(1):
                 x_vision_early = x_vision_early.permute(0, 3, 2, 1)
             x = torch.cat([x, x_vision_early], dim=-1)
        # ----


    
        x = self.FC_1(x)

        # ##### MODIFIE: Calendar / TE Handling #####
        # GMAN expects TE size [B, L + T, 2]
        # We assume x_calendar is provided with at least L steps. 
        # Ideally, x_calendar should cover L + T. If not, we pad or slice.
        if x_calendar is None:
            # x_calendar size: [B,L,2]
            raise NotImplementedError("GMAN requires calendar data (TE). Please provide x_calendar input.")
        if x_calendar.size(-1) != 2:
            raise ValueError(f"Expected x_calendar.size(-1) == 2, but got {x_calendar.size(-1)}. Set args.calendar_types to ['dayofweek', 'timeofday'] and add 'calendar' to dataset_names.")

        # For GMAN to work properly, future time info is ALSO needed.
        current_len = x_calendar.size(1)
        required_len = self.num_his + self.step_ahead//self.horizon_step
        # print('init x_calendar.size(), ', x_calendar.size())  
        if current_len < required_len:
            # The last know time-step is available at the last dimension x_calendar[:,-1,:].
            # Let's infer future time-steps based on the last one, the number of time-steps per day, the step ahead, and the horizon steps:
            last_tod = x_calendar[:, -1, self.pos_tod]  # last time-of-day
            # tod is in pourcentage. 
            def get_marginal_tod_from_increment(last_tod,increment):
                return ((last_tod*self.steps_per_day + (increment * self.horizon_step)) % self.steps_per_day) / self.steps_per_day
            future_tods = [get_marginal_tod_from_increment(last_tod,i) for i in range(1, required_len - current_len + 1)]
            future_tods = torch.stack(future_tods, dim=1)  # [B, required_len - current_len]

            # Future dow such as: 
            # If future_tod < last_tod => dow +1 if dow < 6 else 0
            # if future_tod >= last_tod => dow unchanged
            last_dow = x_calendar[:, -1, self.pos_dow] 
            future_dows = [ (last_dow + ((future_tods[:,i] < last_tod).float())) % 7 for i in range(future_tods.size(1))]
            future_dows = torch.stack(future_dows, dim=1)  # [B, required_len - current_len]
            padding = torch.stack([future_dows, future_tods], dim=-1)  # [B, required_len - current_len, 2]
            x_calendar = torch.cat([x_calendar, padding], dim=1)
            x_calendar[:,:,self.pos_tod] *= self.steps_per_day  # No necessary but just in case, go baxk to integer representation

            # print('x_calendar.size(), ', x_calendar.size())  
            # print('x_calendar[0]', x_calendar[0])
            # print('x_calendar stats: ', torch.min(x_calendar), torch.max(x_calendar), torch.mean(x_calendar))

        elif current_len > required_len:
            x_calendar = x_calendar[:, :required_len, :]

        # STE Generation

        # print('x_calendar.size(), ', x_calendar.size())  
        STE = self.STEmbedding(self.SE, x_calendar)
        STE_his = STE[:, :self.num_his]
        STE_pred = STE[:, self.num_his:]
        
        # Encoder
        for net in self.STAttBlock_1:
            x = net(x, STE_his)
            
        # Transform Attention
        x = self.transformAttention(x, STE_his, STE_pred)
        
        # Decoder
        for net in self.STAttBlock_2:
            x = net(x, STE_pred)

        # ##### MODIFIED: Late Fusion #####
        # x is currently [B, T, N, D]
        late_feats = self.contextual_input_embedding.late_features
        if late_feats.numel() > 0:
             if late_feats.size(1) == self.L and self.L != self.step_ahead:
                late_feats = late_feats[:, -self.step_ahead:, :, :]
             x = self.contextual_input_embedding.concat_features(x, late_feats)
        
        if x_vision_late is not None:
             if x_vision_late.dim() == 4 and x_vision_late.size(1) != x.size(1):
                  x_vision_late = x_vision_late.permute(0, 3, 2, 1)
             x = torch.cat([x, x_vision_late], dim=-1)

        # Output Projection
        x = self.FC_2(x) # [B, T, N, 1]
        del STE, STE_his, STE_pred
        
        # ##### MODIFIE: Final Permute to standard [B, Out_C, N, T] or [B, T, N, 1] -> [B, 1, N, T] #####
        x = x.permute(0, 3, 2, 1) # [B, 1, N, T]
        
        return x



if __name__ == '__main__':
    import argparse
    import numpy as np

    # 1. Configuration et Seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Paramètres de dimension
    B, N, L = 2, 10, 12 # Batch, Nodes, History Length
    T_pred = 12         # Prediction Length (step_ahead)
    C = 1               # Input Channels
    K, d = 2, 4         # Heads configs
    D = K * d

    # 2. Mock des arguments (args)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Args Structurels
    args.num_nodes = N
    args.L = L
    args.step_ahead = T_pred
    args.C = C
    args.bn_decay = 0.1
    args.nb_STAttblocks = 2
    args.num_heads = K
    args.head_dim = d
    
    # Args Temporels & Framework Spécifiques
    args.horizon_step = 1       # Pas de temps entre deux données (souvent 1)
    args.time_step_per_hour = 12 # 12 pas par heure (toutes les 5 min) -> 288 pas/jour
    
    # Configuration des positions contextuelles (Indispensable pour votre logique)
    # Channel 0: DayOfWeek, Channel 1: TimeOfDay
    args.contextual_positions = {
        "calendar_dayofweek": 0, 
        "calendar_timeofday": 1
    }
    
    # Args Fusion (Vides ici)
    args.contextual_kwargs = {}
    args.Early_fusion_names = []
    args.Late_fusion_names = []

    # 3. Génération de Données Cohérentes
    # x : [B, C, N, L] (Format Framework standard avant permutation interne)
    x = torch.randn(B, C, N, L)

    # SE : Spatial Embedding
    SE = torch.randn(N, D)

    # --- GÉNÉRATION CALENDAIRE SPÉCIFIQUE ---
    # Objectif : Tester le "roll-over" (passage minuit) dans votre logique de génération.
    # On génère un historique qui s'arrête juste avant minuit.
    # steps_per_day = 12 * 24 = 288.
    
    # On commence à l'index 280. L'historique sera [280, 281, ..., 288(0), 1...]
    # Si L=12, l'historique va de 280 à 280+11 = 291.
    # 291 % 288 = 3. Donc l'historique traverse minuit.
    
    # Pour bien tester votre logique de *génération future*, on va donner un historique
    # qui s'arrête JUSTE avant minuit, par exemple start=276 (donc fin à 287).
    # Ainsi, la prédiction (future) DEVRA passer à 0 et incrémenter le jour.
    
    start_idx = 276 
    
    # Création des indices temporels pour l'historique (L pas de temps)
    time_indices = np.arange(start_idx, start_idx + L)
    
    # Time of Day (Normalisé entre 0 et 1 comme demandé)
    # tod_val = index % 288 / 288.0
    tod_values = (time_indices % 288) / 288.0
    
    # Day of Week (0-6)
    # On commence jour 0. Si on passe 288, on passe jour 1.
    dow_start = 0 
    dow_values = (dow_start + (time_indices // 288)) % 7
    
    # Construction du tenseur [B, L, 2] -> sera étendu à [B, 2, N, L]
    # Note: On répète les mêmes temps pour tout le batch pour simplifier la vérification visuelle
    calendar_seq = np.stack([dow_values, tod_values], axis=1) # [L, 2]
    calendar_tensor = torch.tensor(calendar_seq, dtype=torch.float32) # [L, 2]
    
    # Adaptation au format d'entrée Framework: [B, C_cal=2, N, L]
    # 1. Unsqueeze Batch: [1, L, 2]
    # 2. Unsqueeze Nodes & Repeat: [1, L, N, 2]
    # 3. Permute vers [1, 2, N, L] (C'est souvent comme ça qu'ils arrivent)
    # 4. Repeat Batch
    x_calendar = calendar_tensor.unsqueeze(0).repeat(B, 1, 1) # [B, L, 2]

    print("-" * 40)
    print("Test GMAN : Vérification Génération Calendaire")
    print("-" * 40)
    print(f"Paramètres: L={L}, Pred={T_pred}, Steps/Day={288}")
    print(f"Input Calendar Length: {x_calendar.shape[-1]} (Historique seul)")
    print(f"Last Input ToD Index (approx): {int(tod_values[-1]*288)}")
    print(f"Last Input DoW: {int(dow_values[-1])}")
    print("-> Le modèle doit générer 12 pas futurs. Le ToD devrait passer à 0 et le DoW incrémenter.")
    
    # 4. Initialisation
    model = GMAN(args, SE=SE)
    model.eval()

    # 5. Exécution
    try:
        with torch.no_grad():
            # On ne passe que l'historique dans x_calendar
            output = model(x, x_calendar=x_calendar, contextual=[])
        
        print("\n✅ SUCCÈS : Le Forward pass s'est terminé sans erreur.")
        print(f"Output Shape: {output.shape} (Attendu: [{B}, 1, {N}, {T_pred}])")
        
        # NOTE: Pour vérifier les valeurs internes générées (ToD/DoW futurs), 
        # il faudrait mettre des prints dans la méthode forward() juste après la génération de 'padding'.
        # Ici, le fait que ça ne crash pas prouve que les dimensions de concaténation étaient correctes.

    except Exception as e:
        print("\n❌ ERREUR lors du Forward pass :")
        print(e)
        import traceback
        traceback.print_exc()

    print("-" * 40)