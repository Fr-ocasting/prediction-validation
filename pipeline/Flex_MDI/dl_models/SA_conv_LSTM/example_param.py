# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_file_path,'..','..'))
if grandparent_dir not in sys.path:
    sys.path.insert(0,grandparent_dir)
# ...

# Personnal import:
from pipeline.Flex_MDI.dl_models.SA_conv_LSTM.constants import WeightsInitializer 

model_params: SAMSeq2SeqParams = {"attention_hidden_dims": 2,
                                    "input_seq_length": input_seq_length,
                                    "num_layers": 2,
                                    "num_kernels": 64,
                                    "return_sequences": False,
                                    "convlstm_params": {"in_channels": 1,
                                                        "out_channels": 1,
                                                        "kernel_size": (3, 3),
                                                        "padding": "same",
                                                        "activation": "relu",
                                                        "frame_size": (64, 64),
                                                        "weights_initializer": WeightsInitializer.He
                                                        }
                                    }


