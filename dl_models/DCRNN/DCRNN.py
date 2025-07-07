import numpy as np
import torch
import torch.nn as nn

# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal import:
from dl_models.DCRNN.dcrnn_cell import DCGRUCell

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get('L'))
        self.seq_len = int(model_kwargs.get('L'))  # for the encoder
        self.device = model_kwargs.get('device')
        self.dropout = model_kwargs.get('dropout')

        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type,device = self.device) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """

        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.device = model_kwargs.get('device')
        # self.output_dim = int(model_kwargs.get('out_dim'))
        self.horizon = int(model_kwargs.get('step_ahead'))  # for the decoder

        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type,device = self.device) for _ in range(self.num_rnn_layers)])
        

        self.out_dim_factor = int(model_kwargs.get('out_dim_factor'))
        self.step_ahead = int(model_kwargs.get('step_ahead'))

        # Tackle Calendar Embedding: 
        if ('calendar_embedding' in model_kwargs.get('dataset_names')):
            self.TE_concatenation_late = model_kwargs.get('args_embedding').concatenation_late 
        else :
            self.TE_concatenation_late = False
        
        if self.TE_concatenation_late:
            self.TE_embedding_dim = model_kwargs.get('args_embedding').embedding_dim 
            self.in_channel_fc1 = self.rnn_units + self.TE_embedding_dim
        else:
            self.in_channel_fc1 = self.rnn_units
        # ...

        self.projection_layer = nn.Linear(self.in_channel_fc1, self.out_dim_factor)


    def forward(self, inputs, hidden_state=None,x_calendar=None):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        reshaped_output = output.view(-1, self.rnn_units)
        # print('\nBefore concatenation: ')
        if self.TE_concatenation_late:
            # print('reshaped_output: ',reshaped_output.size())
            # print('x_calendar: ',x_calendar.size())
            x_calendar = x_calendar.reshape(-1,x_calendar.size(-1)) # [B,C,N,Z]
            reshaped_output = torch.cat([reshaped_output,x_calendar],dim = -1)

        
        # print('reshaped_output: ',reshaped_output.size())
        projected = self.projection_layer(reshaped_output) #[B*N,rnn_unit] ->  [B*N,out_dim_factor]
        # print('projecteded output: ',projected.size())
        output = projected.view(-1, self.num_nodes * self.out_dim_factor)#  [B*N,out_dim_factor] ->  [B,N*out_dim_factor]
        # print('reshaped output: ',output.size())

        return output, torch.stack(hidden_states)


class DCRNN(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, 
                 #logger, 
                 **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.device = model_kwargs.get('device')
        self.encoder_model = EncoderModel(adj_mx, **model_kwargs)
        self.decoder_model = DecoderModel(adj_mx, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False)) 
        self.out_dim_factor =  int(model_kwargs.get('out_dim_factor')) 
        self.step_ahead = int(model_kwargs.get('step_ahead')) 
        self.dropout = model_kwargs.get('dropout')
        #self._logger = logger

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None,x_calendar=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.out_dim_factor),
                                device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                      decoder_hidden_state,
                                                                      x_calendar=x_calendar)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None,x_calendar = None,x_vision = None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)   #i.e [L,B,C*N]
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """ 
        if x_vision is not None:
            raise NotImplementedError('x_vision has not been implemented')
        # Ajout pour matcher mon framework avec le dcrnn :
        if len(inputs.size())<4:
            inputs = inputs.unsqueeze(1)
        B,C,N,L = inputs.size()  
        inputs = inputs.permute(3,0,1,2)  # [B,C,N,L] -> [L,B,C,N]
        inputs = inputs.reshape(inputs.size(0),inputs.size(1),-1)  # [L,B,C,N] -> [L,B,C*N]
        # ...
        encoder_hidden_state = self.encoder(inputs)
        #self._logger.debug("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen,x_calendar=x_calendar)  # outputs -> [B*N, out_dim] 
        #self._logger.debug("Decoder complete")
        #if batches_seen == 0:
        #    self._logger.info(
        #        "Total trainable parameters {}".format(count_parameters(self))
        #    )

        # Ajout pour matcher mon framework avec le dcrnn 
        # Outputs: [out_dim,B,N*out_dim_factor]
        # print('outputs.size: ',outputs.size())
        outputs = outputs.permute(1,0,2)  # [out_dim,B,N*out_dim_factor] -> [B,out_dim,N*out_dim_factor]
        outputs = outputs.reshape(B,self.step_ahead,N,self.out_dim_factor) # [B,out_dim,N*out_dim_factor] -> [B,out_dim,N,out_dim_factor]
        outputs = outputs.permute(0,3,2,1) # [B,out_dim,N,out_dim_factor] -> [B,out_dim_factor,N,out_dim,N]
        # print('outputs.size: ',outputs.size())
        # ...
        return outputs

