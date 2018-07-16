
# coding: utf-8

# In[1]:


import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
import torch

from torch import nn

import torch.functional as F

from torch.optim import Adam
import argparse
import sys
import copy
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from load_data import get_tensor_data_loaders
import numpy as np
import time
import json
from pathlib import Path
from functools import partial
from replay_buffer import ReplayMemory, fill_replay_buffer


# In[2]:


def mkstr(key,args={}):
    d = args.__dict__
    return "=".join([key,str(d[key])])


# In[3]:


def initialize_weights(self):
    # Official init from torch repo.
    for m in self.modules():
        print(m)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()


# In[4]:


class Encoder(nn.Module):
    def __init__(self,in_ch=3,h_ch=32, batch_norm=False):
        super(Encoder,self).__init__()
        bias= False if batch_norm else True
            
        layers = [nn.Conv2d(in_channels=in_ch, out_channels=h_ch,
                      kernel_size=3, stride=2, padding=1,bias=bias),
            nn.BatchNorm2d(h_ch),
            nn.ELU(),
            
            nn.Conv2d(in_channels=h_ch, out_channels=h_ch,
                      kernel_size=3, stride=2, padding=1,bias=bias),
            nn.BatchNorm2d(h_ch),
            nn.ELU(),
            nn.Conv2d(in_channels=h_ch, out_channels=h_ch,
                      kernel_size=3, stride=2, padding=1,bias=bias),
            nn.BatchNorm2d(h_ch),
            nn.ELU(),
            nn.Conv2d(in_channels=h_ch, out_channels=h_ch,
                      kernel_size=3, stride=2, padding=1,bias=bias),
            nn.BatchNorm2d(h_ch),
            nn.ELU()
                 ]
        if not batch_norm:
            for layer in layers:
                if "BatchNorm" in str(layer):
                    layers.remove(layer)
        self.encoder = nn.Sequential(*layers)
                    
        self.fc = nn.Linear(in_features=h_ch, out_features=h_ch)

    def get_output_shape(self,inp_shape):
        a = torch.randn(inp_shape)
        return self.forward(a).size(1)
    
#     def get_feature_maps(self):
#         return self.fmaps

    def forward(self,x):
        fmaps = self.encoder(x)
        vec = fmaps.view(fmaps.size(0),-1)
        self.fmaps = fmaps
        return vec


# In[5]:


# enc = Encoder(batch_norm=True)

# x = torch.randn(8,3,64,64)

# vec = enc(x)

# print(vec.size())
# print(enc.get_output_shape((8,3,64,64)))


# In[6]:


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
        


# In[7]:


# enc = Encoder(batch_norm=True)

# x1 = torch.randn(8,3,64,64)
# x2 = torch.randn(8,3,64,64)
# vec1 = enc(x1)
# vec2 = enc(x2)
# vec = torch.cat((vec1,vec2),dim=-1)
# ap = ActionPredictor(3,1024)

# logits = ap(vec)
# print(logits.size())


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


def write_ims(index,rows,ims,name, iter_):
    num_ims = rows**2
    ims_grid = make_grid((ims.data[index] + 1) / 2, rows)
    writer.add_image(name, ims_grid, iter_)
    


# In[12]:


def save_incorrect_examples(y,action_guess,x0,x1,iter_):
    wrong_actions = y[torch.ne(action_guess,y)].long()
    num_wrong = wrong_actions.size(0)
    right_actions = y[torch.eq(action_guess,y)].long()
    num_right = right_actions.size(0)
    
    if iter_ % 50 == 0:
        try:
            write_ims(ims=x0,index=wrong_actions,rows=int(np.ceil(np.sqrt(num_wrong))),name=mode +"/debug/x0_wrong", iter_=iter_)
            write_ims(ims=x1,index=wrong_actions,rows=int(np.ceil(np.sqrt(num_wrong))),name=mode +"/debug/x1_wrong", iter_=iter_)
        except:
            print("Num wrong and right: ",num_wrong,num_right)
    


# In[13]:


# def save_fmaps(model):
#     fmaps = model.enc.get_feature_maps()
#     return fmaps


# In[14]:


def do_one_iter(x0,x1,y, iter_=0, mode="train"):
    

    if model.training:
        opt.zero_grad()
    output = model(x0,x1)

    loss = criterion(output,y)
    if model.training:
        loss.backward()
        opt.step()

    action_guess = torch.argmax(output,dim=1)
    acc = (float(torch.sum(torch.eq(y,action_guess)).data) / y.size(0))*100
    save_incorrect_examples(y,action_guess,x0,x1,iter_)

    return float(loss.data), acc #, inc_actions.data
    
    


# In[15]:


def do_epoch(dataloader,epoch,mode="train",numiter=-1):
    t0 = time.time()
    mode = "train" if model.training else "val"
    losses = []
    accs = []
    #inc_actions = torch.randn(size=(0,)).to(DEVICE).long()
    for i,(x0,x1,y,r) in enumerate(dataloader):
        x0,x1, y = x0.to(DEVICE), x1.to(DEVICE), y.to(DEVICE)
        #x0,x1 = torch.split(x,dim=2,split_size_or_sections=x.size(3))
        loss,acc = do_one_iter(x0,x1,y,i,mode=mode)
        losses.append(loss)
        accs.append(acc)
        #inc_actions = torch.cat((inc_actions,inc_action))
        if numiter != -1 and i > numiter:
            break
    loss, acc = np.mean(losses), np.mean(accs)
    writer.add_scalar(mode+"/loss",loss,global_step=epoch)
    writer.add_scalar(mode+"/acc",acc,global_step=epoch)

    return loss,acc, time.time() - t0


# In[16]:


def write_to_config_file(dict_,log_dir):
    config_file_path = Path(log_dir) / "config.json"
    dict_string = json.dumps(dict_) + "\n"
    with open(config_file_path, "w") as f:
        f.write(dict_string)
    


# In[17]:


def setup_args():
    tmp_argv = copy.deepcopy(sys.argv)
    test_notebook = False
    if "ipykernel_launcher" in sys.argv[0]:
        sys.argv = [""]
        test_notebook= True
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--env_name",type=str, default='MiniGrid-Empty-6x6-v0'),
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--resize_to",type=int, nargs=2, default=[42, 42])
    parser.add_argument("--epochs",type=int,default=100000)
    parser.add_argument("--dataset_size",type=int,default=60000)
    parser.add_argument("--width",type=int,default=32)
    #parser.add_argument("--grayscale",action="store_true")
    parser.add_argument("--batch_norm",action="store_true")
    parser.add_argument("--offline",action="store_true")
    parser.add_argument("--buffer_size",type=int,default=10**6)
    parser.add_argument("--action_strings",type=str, nargs='+', default=["move_up", "move_down", "move_right", "move_left"])
    args = parser.parse_args()
    args.resize_to = tuple(args.resize_to)
    if args.batch_size > args.dataset_size:
        args.batch_size = args.dataset_size
    sys.argv = tmp_argv
    if test_notebook:
        args.dataset_size = 1000
        #args.online=True
    mstr = partial(mkstr,args=args)
    output_dirname = ("notebook_" if test_notebook else "") + "_".join([mstr("env_name"),
                                                                        mstr("lr"),
                                                                        mstr("width"),
                                                                        mstr("dataset_size"),
                                                                        mstr("resize_to")
                                                                       ])
    args.output_dirname = output_dirname
    return args


# In[18]:


def setup_dirs_logs(args):
    log_dir = './.logs/%s'%args.output_dirname
    writer = SummaryWriter(log_dir=log_dir)
    write_to_config_file(args.__dict__, log_dir)

    return writer


# In[19]:


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    args = setup_args()
    writer = setup_dirs_logs(args)

    if args.offline:
        trl, vall, tel, label_list = get_tensor_data_loaders(rollout_size=128,action_strings=args.action_strings,
                                                         env_name=args.env_name, resize_to = args.resize_to,
                                                        batch_size = args.batch_size, total_examples=args.dataset_size)
    else:
        trl = ReplayMemory(capacity=args.buffer_size,batch_size=args.batch_size)
        trl, label_list = fill_replay_buffer(trl,int(0.8*args.dataset_size),rollout_size=128,action_strings=args.action_strings,
                                                         env_name=args.env_name, resize_to = args.resize_to)
        vall = ReplayMemory(capacity=args.buffer_size,batch_size=args.batch_size)
        vall, _ = fill_replay_buffer(vall,int(0.1*args.dataset_size),rollout_size=128,action_strings=args.action_strings,
                                                         env_name=args.env_name, resize_to = args.resize_to)


# In[26]:



num_actions = len(label_list)
in_ch = trl.dataset.tensors[0].size()[1] if args.offline else trl.memory[0].next_state.shape[2]
model = InverseModel(in_ch=in_ch,im_wh=args.resize_to,h_ch=args.width,
                     num_actions=num_actions,batch_norm=args.batch_norm).to(DEVICE)
_ = model.apply(initialize_weights)
opt = Adam(lr=args.lr, params=model.parameters())
criterion = nn.CrossEntropyLoss()

if not args.offline:
    numiter = int(args.dataset_size / args.batch_size)
else:
    numiter = -1
for epoch in range(args.epochs):
    model.train()
    print("Beginning epoch %i"%(epoch))
    loss,acc,t = do_epoch(trl,epoch,mode="train",numiter=numiter)
    print("\tTr Time: %8.4f seconds"% (t))
    print("\tTr Loss: %8.4f \n\tTr Acc: %9.3f%%"%(loss,acc))
    
    model.eval()
    vloss,vacc,t = do_epoch(vall,epoch,mode="val",numiter=numiter)
    print("\n\tVal Time: %8.4f seconds"% (t))
    print("\tVal Loss: %8.4f \n\tVal Acc: %9.3f %%"%(vloss,vacc))
    


