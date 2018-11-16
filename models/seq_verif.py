#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import torch.functional as F

from models.base_encoder import Encoder
from utils import classification_acc
import numpy as np

class InOrderBinaryClassifier(nn.Module):
    def __init__(self,in_ch, h_ch=256):
        super(InOrderBinaryClassifier,self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_features=in_ch,out_features=h_ch),
            nn.ReLU(),
            nn.Linear(in_features=h_ch,out_features=2)
        )
    def forward(self,x):
        return self.predictor(x)
        

class ShuffleNLearn(nn.Module):
    def __init__(self, num_frames=3, embed_len=32, **kwargs):
        super(ShuffleNLearn,self).__init__()
        self.embed_len = embed_len
        self.encoder = Encoder(embed_len = embed_len, **kwargs)
        self.bin_clsf = InOrderBinaryClassifier(in_ch=num_frames*self.embed_len)
    
    def forward(self,xs):
        f = torch.cat([self.encoder(x) for x in xs])
        return self.bin_clsf(f)
    
    def shuffle(self,xs):
        a,b,c,d,e = [xs[:,i] for i in range(5)]
        bcd = copy.deepcopy(torch.cat(b,c,d))
        bad = copy.deepcopy(torch.cat(b,a,d))
        bed = copy.deepcopy(torch.cat(b,e,d))

        
    def loss_acc(self, trans):
        xs = copy.deepcopy(trans.xs)
        x_shuff, true = self.shuffle(xs)
        pred = self.forward(x_shuff)
        acc = classification_acc(logits=pred,true=true)
        loss = nn.CrossEntropyLoss()(pred,true)
        return loss, acc


if __name__ == "__main__":
    from torchvision.utils import make_grid
    %matplotlib inline
    from matplotlib import pyplot as plt

    tr, val = bufs
    batch_size = 32

    trans = tr.sample(batch_size)

    xs = trans.xs

    xs.shape

    a,b,c,d,e,f,g,h,i,j = [xs[:,i] for i in range(10)]

    bcd = copy.deepcopy(torch.stack((d,e,f)))
    bad = copy.deepcopy(torch.stack((d,a,f)))
    bed = copy.deepcopy(torch.stack((d,j,f)))

    # bcd = copy.deepcopy(torch.stack((b,c,d)))
    # bad = copy.deepcopy(torch.stack((b,a,d)))
    # bed = copy.deepcopy(torch.stack((b,e,d)))

    bcdbadbed = torch.stack((bcd,bad,bed))

    bcdbadbed.shape

    probs = torch.tensor([0.5,0.25,0.25])

    inds = torch.multinomial(input=probs, num_samples=batch_size, replacement=True)

    shuffled_batch = torch.stack([bcdbadbed[inds[i],:,i] for i in range(batch_size) ])

    shuffled_batch.shape

    for i,tims in enumerate(shuffled_batch):
        t = make_grid(tims,3,normalize=True, pad_value=0.2).detach().cpu().numpy()
        im = t.transpose(1,2,0)

        title = ["bcd","bad", "bed"][inds[i]]

        plt.figure(i)

        plt.title(title)

        plt.imshow(im)