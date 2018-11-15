import torch
from torch import nn
import torch.functional as F
from models.base_encoder import Encoder
from evaluations.utils import classification_acc

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
    def __init__(self, embed_len=32, num_actions=3, **kwargs):
        super(InverseModel,self).__init__()
        self.embed_len = embed_len
        self.encoder = Encoder(embed_len=embed_len, **kwargs)
        self.ap = ActionPredictor(num_actions=num_actions, in_ch=2*self.embed_len)
    
    def forward(self,xs):
        f0 = self.encoder(xs[:,0])
        f1 = self.encoder(xs[:,1])
        fboth = torch.cat([f0,f1],dim=-1)
        return self.ap(fboth)
    
    def loss_acc(self, trans):
        # for now just select the first two frames (but maybe in the future we could select every possible frame
        # or some combination of them)
        
        pred = self.forward(trans.xs)
        true = trans.actions[:,0]
        # print(pred.size())
        # print(true.size())
        acc = classification_acc(logits=pred,true=true)
        loss = nn.CrossEntropyLoss()(pred,true)
        return loss, acc
    