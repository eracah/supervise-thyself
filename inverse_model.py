
# coding: utf-8

# In[1]:


import torch

from torch import nn

import torch.functional as F

from torch.optim import Adam
import argparse
import sys
import copy
from tensorboardX import SummaryWriter
from data import make_batch
from torchvision.utils import make_grid
from data import get_data_loaders
import numpy as np
import time


# In[2]:


def mkstr(key):
    d = args.__dict__
    return "=".join([key,str(d[key])])


# In[3]:


class Encoder(nn.Module):
    def __init__(self,in_ch=3,h_ch=32, batch_norm=False):
        super(Encoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=h_ch,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(h_ch),
            nn.ELU(),
            
            nn.Conv2d(in_channels=h_ch, out_channels=h_ch,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(h_ch),
            nn.ELU(),
            nn.Conv2d(in_channels=h_ch, out_channels=h_ch,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(h_ch),
            nn.ELU(),
            nn.Conv2d(in_channels=h_ch, out_channels=h_ch,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(h_ch),
            nn.ELU()
        )
        self.fc = nn.Linear(in_features=h_ch,out_features=h_ch)
        if not batch_norm:
            del self.encoder[1]
    def get_output_shape(self,inp_shape):
        a = torch.randn(inp_shape)
        return self.forward(a).size(1)
    def forward(self,x):
        fmaps = self.encoder(x)
        vec = fmaps.view(fmaps.size(0),-1)
        return vec


# In[4]:


enc = Encoder(batch_norm=True)

x = torch.randn(8,3,64,64)

vec = enc(x)


# In[5]:


print(vec.size())
print(enc.get_output_shape((8,3,64,64)))


# In[6]:


class ActionPredictor(nn.Module):
    def __init__(self, num_actions, in_ch, h_ch=256):
        super(ActionPredictor,self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_features=in_ch,out_features=h_ch),
            nn.Linear(in_features=h_ch,out_features=num_actions)
        )
    def forward(self,x):
        return self.predictor(x)
        


# In[7]:


enc = Encoder(batch_norm=True)

x1 = torch.randn(8,3,64,64)
x2 = torch.randn(8,3,64,64)
vec1 = enc(x1)
vec2 = enc(x2)
vec = torch.cat((vec1,vec2),dim=-1)
ap = ActionPredictor(3,1024)

logits = ap(vec)
print(logits.size())


# In[8]:


class InverseModel(nn.Module):
    def __init__(self,in_ch,im_wh, h_ch, num_actions, batch_norm):
        super(InverseModel,self).__init__()
        self.enc = Encoder(in_ch=in_ch,h_ch=h_ch, batch_norm=batch_norm)
        
        embed_len = self.enc.get_output_shape((1, in_ch, *im_wh))
        self.ap = ActionPredictor(num_actions=num_actions,in_ch=2*embed_len)
    def forward(self,x0,x1):
        f0 = self.enc(x0)
        f1 = self.enc(x1)
        fboth = torch.cat([f0,f1],dim=-1)
        return self.ap(fboth)


# In[9]:


# prd = InverseModel(in_ch=3,im_wh=(64,64),h_ch=32,num_actions=4,batch_norm=False)

# x1 = torch.randn(8,3,64,64)
# x2 = torch.randn(8,3,64,64)

# prd(x1,x2)


# In[10]:


# prd.parameters()

# nn.CrossEntropyLoss?


# In[11]:


def do_one_iter(x,y, iter_=0, mode="train"):
    x0,x1 = torch.split(x,dim=2,split_size_or_sections=x.size(3))


   
    if iter_ % 50 == 0:
        rows = 6 
        num_ims = rows**2
        x0_grid = make_grid((x0.data[:num_ims] + 1) / 2, rows)
        writer.add_image(mode + "/x0", x0_grid, iter_)
        x1_grid = make_grid((x1.data[:num_ims] +1)/2,rows)
        writer.add_image(mode + "/x1", x1_grid, iter_)

    if model.training:
        opt.zero_grad()
    output = model(x0,x1)

    loss = criterion(output,y)
    if model.training:
        loss.backward()
        opt.step()

    action_guess = torch.argmax(output,dim=1)
    acc = (float(torch.sum(torch.eq(y,action_guess)).data) / y.size(0))*100
    return float(loss.data), acc
    
    


# In[12]:


def do_epoch(dataloader,epoch,mode="train",numiter=-1):
    t0 = time.time()
    mode = "train" if model.training else "val"
    losses = []
    accs = []
    for i,(x,y) in enumerate(dataloader):
        x,y = x.to(DEVICE), y.to(DEVICE)
        loss,acc = do_one_iter(x,y,i,mode=mode)
        losses.append(loss)
        accs.append(acc)
        if numiter != -1 and i > numiter:
            break
    loss, acc = np.mean(losses), np.mean(accs)
    writer.add_scalar(mode+"/loss",loss,global_step=epoch)
    writer.add_scalar(mode+"/acc",acc,global_step=epoch)
    
    return loss,acc, time.time() - t0


# In[ ]:


if __name__ == "__main__":
    tmp_argv = copy.deepcopy(sys.argv)
    test_notebook = False
    if "ipykernel_launcher" in sys.argv[0]:
        sys.argv = [""]
        test_notebook= True
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--env_name",type=str, default='MiniGrid-Empty-6x6-v0'),
    parser.add_argument("--batch_size",type=int,default=128)
    #parser.add_argument("--resize_to",type=int, nargs=2, default=[-1, -1])
    parser.add_argument("--epochs",type=int,default=100000)
    parser.add_argument("--data_dir",type=str,default="../data")
    args = parser.parse_args()
    #args.resize_to = tuple(args.resize_to)
    sys.argv = tmp_argv

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



    output_dirname = "_".join([mkstr("env_name"),mkstr("lr")])
    if test_notebook:
        output_dirname = "notebook_" + output_dirname
    log_dir = './.logs/%s'%output_dirname

    writer = SummaryWriter(log_dir=log_dir)

    trl, vall, tel = get_data_loaders(batch_size=args.batch_size)
    num_actions = 3
    model = InverseModel(in_ch=3,im_wh=(64,64),h_ch=128,num_actions=num_actions,batch_norm=True).to(DEVICE)
    opt = Adam(lr=args.lr, params=model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        print("Beginning epoch %i"%(epoch))
        loss,acc,t = do_epoch(trl,epoch,mode="train")
        print("\tTr Time: %8.4f seconds"% (t))
        print("\tTr Loss: %8.4f \n\tTr Acc: %9.3f%%"%(loss,acc))

        model.eval()
        vloss,vacc,t = do_epoch(vall,epoch,mode="val")
        print("\n\tVal Time: %8.4f seconds"% (t))
        print("\tVal Loss: %8.4f \n\tVal Acc: %9.3f %%"%(loss,acc))
    

