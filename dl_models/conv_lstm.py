import torch
import torch.nn as nn 

# ====================================================================================
# Original Author: https://github.com/NVlabs/conv-tt-lstm/
# 2020 NVIDIA Corportation 
# ================================================================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size = 3, biais = True):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        kernel_size = (kernel_size,kernel_size)

        padding = kernel_size[0] // 2, kernel_size[1]//2
        self.conv = nn.Conv2d(in_channels=input_channels+hidden_channels,
                              out_channels= 4*hidden_channels,
                              kernel_size=kernel_size,
                              padding= padding,
                              biais = biais
                              )
        
        # Init
        self.hidden_states, self.cell_state = None, None

    def initialize(self,inputs):
        '''
        Initialization of Convolution-LSTM cell

        Args: 
        --------
        inputs : a 4-th order tensor of size [B,C,H,W]
        '''
        B,C,H,W = inputs.size()
        self.hidden_states = torch.zeros(B, self.hidden_channels, H, W, device = inputs.device)
        self.cell_states = torch.zeros(B, self.hidden_channels, H, W, device = inputs.device)

    def foward(self, inputs, first_step = False):
        '''
        Args 
        --------
        inputs : a 4-th order tensor of size [B,C,H,W]

        Returns: 
        --------
        hidden_states: a 4-th order tensor of size [B,C',H,W]
        '''
        if first_step: self.initialize(inputs)

        # [B,C,H,W], [B,C',H,W] ->  [B,C+C',H,W]
        concat_inputs_hidden = torch.cat([inputs, self.hidden_states], dim = 1)
        
        # [B,C+C',H,W]-> [B,4*C',H',W']
        concat_conv = self.conv(concat_inputs_hidden)

        # [B,4*C',H',W']-> [B,C',H',W'],[B,C',H',W'],[B,C',H',W'],[B,C',H',W']
        cc_i, cc_f, cc_o, cc_g = torch.split(concat_conv, self.hidden_channels, dim = 1)

        # LSTM -> [B,C',H',W']
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        self.cell_states   = f * self.cell_states + i * g
        self.hidden_states = o * torch.tanh(self.cell_states)
        # ....
        
        return self.hidden_states 
