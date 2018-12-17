import torch
from torch import nn
import torch.functional as F
from models.base_encoder import Encoder
from evaluations.utils import classification_acc
from models.base_linear_model import LinearModel


class InverseModel(nn.Module):
    def __init__(self, embed_len=32, num_actions=3, **kwargs):
        super(InverseModel,self).__init__()
        self.embed_len = embed_len
        self.encoder = Encoder(embed_len=embed_len, **kwargs)
        self.action_predictor = LinearModel(in_feat=2*self.embed_len,
                                            out_feat=num_actions)
    
    def forward(self,xs):
        f0 = self.encoder(xs[:,0])
        f1 = self.encoder(xs[:,1])
        fboth = torch.cat([f0,f1],dim=-1)
        return self.action_predictor(fboth)
    
    def loss_acc(self, trans):
        # for now just select the first two frames (but maybe in the future we could select every possible frame
        # or some combination of them)        
        pred = self.forward(trans.xs)
        true = trans.actions[:,0]
        acc = classification_acc(logits=pred,true=true)
        loss = nn.CrossEntropyLoss()(pred,true)
        return loss, acc
    