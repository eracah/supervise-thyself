import torch
from torch import nn
import torch.functional as F
from models.base_encoder import Encoder

class ActionPredictor(nn.Module):
    def __init__(self, num_actions, in_ch, h_ch=256):
        super(ActionPredictor,self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_features=in_ch,out_features=h_ch),
            nn.ReLU(),
            nn.Linear(in_features=h_ch,out_features=num_actions)
        )
    def forward(self,x):
        return self.predictor(x)
        

class InverseModel(nn.Module):
    def __init__(self, in_ch=3, im_wh=(64,64), h_ch=32, embed_len=32, batch_norm=False, num_actions=3):
        super(InverseModel,self).__init__()
        self.encoder = Encoder(in_ch=in_ch, im_wh=im_wh, h_ch=h_ch, embed_len=embed_len, batch_norm=batch_norm)
        self.ap = ActionPredictor(num_actions=num_actions,in_ch=2*self.encoder.embed_len)
    
    def forward(self,x0,x1):
        f0 = self.encoder(x0)
        f1 = self.encoder(x1)
        fboth = torch.cat([f0,f1],dim=-1)
        return self.ap(fboth)
    
    def loss(self,pred, true):
        return nn.CrossEntropyLoss()(pred,true)
