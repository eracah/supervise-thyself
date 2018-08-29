
# coding: utf-8

# In[2]:


import custom_grids
import sys
from torch import nn
import torch
from torchvision.utils import make_grid
import numpy as np
import json
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from matplotlib import pyplot as plt
from gym_minigrid.wrappers import *
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, Grayscale
import numpy as np
from copy import deepcopy
from functools import partial
import math
from pathlib import Path
from collections import namedtuple
from itertools import product
import random
from tensorboardX import SummaryWriter


# In[3]:


def setup_dirs_logs(args, exp_name):
    base = Path("./.results")
    exp_dir = base / exp_name
    writer = SummaryWriter(log_dir=str(exp_dir))
    write_to_config_file(args.__dict__, exp_dir)
    model_dir = exp_dir / Path("models")
    model_dir.mkdir(exist_ok=True,parents=True)
    return writer, model_dir


# In[4]:


def parse_minigrid_env_name(name):
    return name.split("-")[2].split("x")[0]


# In[5]:


def setup_env(env_name):
    env = gym.make(env_name)
    if "MiniGrid" in env_name:
        action_space = range(3)
        grid_size = env.grid_size - 2
        num_directions = 4
        tot_examples = grid_size**2 * num_directions * len(action_space)
    else:
        action_space = list(range(env.action_space.n))
        grid_size = None
        num_directions = None
        tot_exampls = None
    num_actions = len(action_space)
    return env, action_space, grid_size, num_directions, tot_examples


# In[6]:


def bin_direction(direction):
    """takes 2D unit vector of direction and bins it to 4 directions right,down,left,up
       0: right
       1: down
       2: left
       3: up """
    
    minigrid_directions = [[1, 0],[ 0, -1],[-1 , 0],[0, 1]]
    binned_direction = minigrid_directions.index(list(direction))
    return binned_direction


# In[7]:


def unbin_direction(binned_direction):
    """
       0: right
       1: down
       2: left
       3: up """
    index = binned_direction
    minigrid_directions = [[1, 0],[ 0, -1],[-1 , 0],[0, 1]]
    direction_vector = minigrid_directions[index]
    return direction_vector
    


# In[8]:


def classification_acc(logits,true):
    guess = torch.argmax(logits,dim=1)
    acc = (float(torch.sum(torch.eq(true,guess)).data) / true.size(0))
    return acc


# In[9]:



def convert_frame(obs, resize_to=(84,84),to_tensor=False):
    pil_image = Image.fromarray(obs, 'RGB')
    
    transforms = [Resize(resize_to)] if resize_to != (-1,-1) else []
    if to_tensor:
        transforms.extend([ToTensor(),Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transforms = Compose(transforms)
    frame = transforms(pil_image)
    if to_tensor:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        frame= frame.to(DEVICE)
    else:
        frame = np.asarray(frame)


    
    return frame

def convert_frames(frames,resize_to=(64,64),to_tensor=False):
    convert = partial(convert_frame,resize_to=resize_to,to_tensor=to_tensor)
    return torch.stack([convert(frame) for frame in frames])
        


# In[10]:


def get_trans_tuple():
        tuple_fields = ['x0','x1','a', 'r', 'done']
        

        tuple_fields.extend(['x0_coord_x', 'x0_coord_y', 'x1_coord_x', 'x1_coord_y'])

        tuple_fields.extend(["x0_direction","x1_direction"])
        
        Transition = namedtuple("Transition",tuple(tuple_fields))
        return Transition
        


# In[18]:


def mkstr(key,args={}):
    d = args.__dict__
    return "=".join([key,str(d[key])])


# In[19]:


def initialize_weights(self):
    # Official init from torch repo.
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()


# In[20]:


def write_ims(index,rows,ims,name, iter_):
    num_ims = rows**2
    ims_grid = make_grid((ims.data[index] + 1) / 2, rows)
    writer.add_image(name, ims_grid, iter_)
    


# In[21]:


def write_to_config_file(dict_,log_dir):
    config_file_path = Path(log_dir) / "config.json"
    dict_string = json.dumps(dict_) + "\n"
    with open(config_file_path, "w") as f:
        f.write(dict_string)
    


# In[22]:


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


# In[19]:


def plot_test(x0s,x1s,ys,rs, label_list):
    def plot(x0,x1,y,i):
        from matplotlib import pyplot as plt
        get_ipython().run_line_magic('matplotlib', 'inline')
        plt.figure(i)
        plt.clf()
        sp1 = plt.subplot(1,2,1)
        sp1.imshow(x0[0])
        sp2 = plt.subplot(1,2,2)
        sp2.imshow(x1[0])
        plt.title(label_list[y])
    if len(x0s.size()) == 3:
        plot(x0s,x1s,ys,i=1)
    else:
        for i in range(x0s.size()[0]):
            plot(x0s[i],x1s[i],ys[i],i)
    plt.show()


# In[2]:


# if __name__ == "__main__":
#     from replay_buffer import create_and_fill_replay_buffer
    
#     env = gym.make("MiniGrid-Empty-6x6-v0")
#     rb = create_and_fill_replay_buffer(env=env,size=10, 
#                                      other_buffers=[])

#     used = set([(t.x0_coord_x,t.x0_coord_y,t.a,t.x0_direction) for t in rb.memory])

#     unused = []
#     for t in unused_datapoints_iterator(other_buffer=rb,env=env ):
#         unused.append((t.x0_coord_x,t.x0_coord_y,t.a,t.x0_direction))
#     unused_set = set(unused)

#     assert used.isdisjoint(unused_set)
#     assert len(used.union(unused_set)) == (env.grid_size - 2)**2 * 3 * 4


# In[ ]:


#directions = [[1, 0],[ 0, -1],[-1 , 0],[0, 1]]

# ind_to_str = dict(zip(range(4),["right","down","left","up"]))

# env = gym.make("MiniGrid-Empty-6x6-v0")

# _=env.reset()
# #print(env.get_dir_vec())
# for i in range(4):
#     direc = list(env.get_dir_vec())
#     direc_label = dirs.index(direc)
#     print(direc,direc_label, ind_to_str[direc_label])
#     _ = env.step(0)


