import torch
from torch import nn
import torch.functional as F
import copy
from models.base_encoder import Encoder
from evaluations.utils import classification_acc
import numpy as np
from models.base_linear_model import LinearModel


class ShuffleNLearn(nn.Module):
    def __init__(self, num_frames=3, embed_len=32, **kwargs):
        super(ShuffleNLearn,self).__init__()
        self.embed_len = embed_len
        self.encoder = Encoder(embed_len = embed_len, **kwargs)
        self.bin_clsf = LinearModel(in_feat=num_frames*self.embed_len, out_feat=2)
        self.args = kwargs["args"]
        self.stride = self.args.stride
        self.num_frames = self.args.frames_per_example
    
    def forward(self,xs):
        f = torch.cat([self.encoder(xs[:,i]) for i in range(xs.shape[1])], dim=1)
        return self.bin_clsf(f)
    
    def shuffle(self,xs):
        batch_size, num_frames_per_example = xs.shape[0], xs.shape[1]
        assert xs.shape[1] >= 5
        ind = torch.linspace(0,xs.shape[1] - 1,steps=5).round().long()
        x_subsampled = torch.index_select(input=xs,dim=1,index=ind)
        a,b,c,d,e = [x_subsampled[:,i] for i in range(5)]
        bcd = copy.deepcopy(torch.stack((b,c,d)))
        bad = copy.deepcopy(torch.stack((b,a,d)))
        bed = copy.deepcopy(torch.stack((b,e,d)))
        bcdbadbed = torch.stack((bcd,bad,bed))
        probs = torch.tensor([0.5,0.25,0.25])
        inds = torch.multinomial(input=probs, num_samples=batch_size, replacement=True)
        shuffled_batch = torch.stack([bcdbadbed[inds[i],:,i] for i in range(batch_size) ])
        
        device = self.args.device  #encoder.fc.weight.device.type
        true = (inds < 1).long().to(device) #bcd (0) is correct ordering bad and bed  (1,2) are incorrect
        return shuffled_batch, true

        
    def loss_acc(self, trans):
        xs = copy.deepcopy(trans.xs)
        x_shuff, true = self.shuffle(xs)
        pred = self.forward(x_shuff)
        acc = classification_acc(logits=pred,true=true)
        loss = nn.CrossEntropyLoss()(pred,true)
        return loss, acc


