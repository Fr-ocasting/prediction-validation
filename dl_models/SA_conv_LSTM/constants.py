import torch
from typing import NotRequired, Tuple, TypedDict, Union
from typing import TypedDict
from enum import Enum

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WeightsInitializer(str, Enum):
    Zeros = "zeros"
    He = "he"
    Xavier = "xavier"


class ConvLSTMParams(TypedDict):
    in_channels: int
    out_channels: int
    kernel_size: Union[int, Tuple]
    padding: Union[int, Tuple, str]
    activation: str
    frame_size: Tuple[int, int]
    weights_initializer: NotRequired[WeightsInitializer]

class SAMConvLSTMParams(TypedDict):
    attention_hidden_dims: int
    convlstm_params: ConvLSTMParams