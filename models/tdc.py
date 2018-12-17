import torch
from torch import nn
import torch.functional as F
from models.base_encoder import Encoder
from evaluations.utils import classification_acc
from models.base_linear_model import LinearModel
import numpy as np
import random

class TDC(nn.Module):
    def __init__(self, embed_len=32, **kwargs):
        super(TDC,self).__init__()
        self.args = kwargs["args"]
        self.embed_len = embed_len
        self.encoder = Encoder(embed_len=embed_len, **kwargs)
        self.interval_choices = [[0],[1],[2],[3,4],list(range(5,10))]
        self.num_buckets = len(self.interval_choices)
        self.temp_dist_predictor = LinearModel(in_feat=2*self.embed_len,
                                            out_feat=self.num_buckets)
    
    def forward(self,xs):
        x0,xt, interval_index = self.pick_frames(xs)
        f0 = self.encoder(x0)
        ft = self.encoder(xt)
        fboth = torch.cat([f0,ft],dim=-1)
        pred = self.temp_dist_predictor(fboth)
        true = interval_index
        return pred,true
    
    def pick_frames(self,xs):
        batch_size = xs.shape[0]
        
        interval_inds = random.choices(range(len(self.interval_choices)), k=batch_size)
        intervals = [self.interval_choices[i] for i in interval_inds]
        dts = [random.choice(interval) for interval in intervals]
        x0 = xs[:,0]
        xt = torch.stack([xs[i,dts[i],:] for i in range(batch_size)])
        return x0,xt, torch.tensor(interval_inds).to(self.args.device)
        
    def loss_acc(self, trans):   
        pred, true = self.forward(trans.xs)
        acc = classification_acc(logits=pred,true=true)
        loss = nn.CrossEntropyLoss()(pred,true)
        return loss, acc
    