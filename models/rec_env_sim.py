from vae import Decoder
from base_encoder import Encoder
from torch import nn


class RecEnvSimulator(nn.Module):
    def __init__(self, embed_len=32, **kwargs):
        super(RecEnvSimulator,self).__init__()