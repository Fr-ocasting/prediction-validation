from SA_conv_LSTM.constants import WeightsInitializer 

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


