import torch
from torch import nn
import torch.functional as F
import numpy as np
from models.base_encoder import Encoder

class RawPixelsEncoder(nn.Module):
    def __init__(self, im_wh=(64,64),in_ch=3, **kwargs):
        super(RawPixelsEncoder,self).__init__()
        self.embed_len = np.prod(im_wh) * in_ch
    def forward(self,x):
        return x.view(x.size(0),-1)

class RandomLinearProjection(nn.Module):
    def __init__(self,im_wh=(64,64),in_ch=3, embed_len=32, **kwargs):
        super(RandomLinearProjection,self).__init__()
        self.embed_len = embed_len
        self.input_len = np.prod(im_wh) * in_ch
        self.fc = nn.Linear(in_features=self.input_len,out_features=self.embed_len)
    def forward(self,x):
        vec = x.view(x.size(0),-1)
        return self.fc(vec)
        

class RandomWeightCNN(Encoder):
    def __init__(self, **kwargs):
        super(RandomWeightCNN,self).__init__(**kwargs)
