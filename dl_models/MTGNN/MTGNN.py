import torch.nn as nn
import torch
import torch.nn.functional as F

# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal import:
from dl_models.MTGNN.MTGNN_layer import graph_constructor, LayerNorm,dilated_inception,mixprop

# Input : 
#   - Semble travailler avec un X [B,C,N,L]


# Si gcn_true :
#   - On fait des graph convolution 
#   - Et on utilise 'adp' (la matrice d'adjacnece). On se sert de 'graph_constructor' si elle n'est pas déjà définie

# A chaque sortie, une layer norm  

# A chaque couche, ajout d'un "filter_conv" et d'une "gate_conv", qui est issu de 'dilated inception'. 
#   - Le filtre passe par une tanh
#   - La gate passe par une sigmoid
#   - Les deux sont multiplié, puis passe dans un dropout
class MTGNN(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, n_vertex, device, 
                 predefined_A=None, static_feat=None, dropout=0.3, 
                 subgraph_size=20, 
                 node_dim=40, 
                 dilation_exponential=1, 
                 conv_channels=32, 
                 residual_channels=32, 
                 skip_channels=64, end_channels=128, 
                 seq_length=12, c_in=2, out_dim=12, 
                 layers=3, propalpha=0.05, 
                 tanhalpha=3, 
                 layer_norm_affline=True,L_add=0,
                 vision_concatenation_late = False,TE_concatenation_late = False,vision_out_dim = None,TE_embedding_dim = None
                 ):
        super(MTGNN, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.n_vertex = n_vertex
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.out_dim = out_dim
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=c_in,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(n_vertex, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        if L_add != 0:
            self.seq_length = seq_length + L_add
        else:
            self.seq_length = seq_length
            
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))    # Concatène plusieurs temporal conv 
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))      # Concatène plusieurs temporal conv 
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, n_vertex, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, n_vertex, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers

        ## ======= Tackle Output Module if concatenation with contextual data: 
        in_channels_end_conv_1 = skip_channels
        self.vision_concatenation_late = vision_concatenation_late
        self.TE_concatenation_late = TE_concatenation_late
        if self.vision_concatenation_late:
            in_channels_end_conv_1 = in_channels_end_conv_1+ vision_out_dim
        if self.TE_concatenation_late:
            in_channels_end_conv_1 = in_channels_end_conv_1+ TE_embedding_dim
        ## =======

        self.end_conv_1 = nn.Conv2d(in_channels=in_channels_end_conv_1,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=c_in, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=c_in, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.n_vertex).to(device)


    def forward(self, x,x_vision=None,x_calendar = None,idx=None):
        '''
        Inputs: 
        --------
        x:  [B,C,N,L]

        Outputs:

        '''
        # Vérifie que x:  [B,C,N,L], et met le 'padding' necessaire pour pouvoir faire les temporal conv avec dilation
        if len(x.size())<4:
            x = x.unsqueeze(1)
            
        B,C,N,L = x.size() 
        assert L==self.seq_length, f'input sequence length {L} not equal to preset sequence length {self.seq_length}'

        if self.seq_length<self.receptive_field:
            x = nn.functional.pad(x,(self.receptive_field-self.seq_length,0,0,0))
        # ...

        # Load Adjacency Matrix si on compte faire de la Graph-convolution
        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A
        # ...
                
        # Skip connection sur l'Input. Conv2D avec gros nombre de channel, avec un grand kernel (tout le long de la séquence)
        skip = self.skip0(F.dropout(x, self.dropout, training=self.training))

        # Embedding avec une conv2D (1,1) pour avoir C = 32
        x = self.start_conv(x)


        # Avant les couches: x.shape : [B,residual channel, N, receptive_field]
        # Parcours le long des couches. Dernière dim passe de L (ou receptive_field) à 1
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            # Graph conv si necessaire, sinon 
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)
        # Après les couches: x.shape : [B,residual channel, N, 1]


        # Skip de l'Input + Embedding  (conv2D kernel (1,1) si sequence petite ou (1,s) avec L sequence grande) de l'output des couches 
        skip = self.skipE(x) + skip

        ## == Concatenation of Contextual Data Before output Module :
        # skip size: [B,H,N,1]
        if self.vision_concatenation_late:
            # [B,1,N,Z] -> [B,Z,N,1]     (1 = C_out actually )
            x_vision = x_vision.permute(0,3,2,1)
            # Concat [B,H,N, 1] + [B,Z,N,1] ->  [B,H',N, 1]
            if not (x.numel() == 0):
                skip = torch.concat([skip,x_vision],axis=1)
            else:
                skip = x_vision
        if self.TE_concatenation_late:
            # [B,1,N,L_calendar]  -> [B,L_calendar,N,1] 
            x_calendar = x_calendar.permute(0,3,2,1)
            # Concat  [B,H,N,1] +  [B,L_calendar,N,1]  ->  [B,H',N, 1]
            if not (x.numel() == 0):
                skip = torch.concat([skip,x_calendar],axis=1)
            else:
                skip = x_calendar
        ## == ...

        # Sortie embedding et Relu en Série. Après les end_conv : [B,1, N, 1] (car out_dim =1 ???)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        # [B,1, N, 1] -> [B,N]
        x = x.squeeze()
        if self.out_dim >1:
            x= x.permute(0,2,1)
        else: 
            x = x.unsqueeze(-1)
        return x