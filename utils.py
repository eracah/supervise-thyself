
# coding: utf-8

# In[1]:


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


# In[ ]:


def parse_minigrid_env_name(name):
    return name.split("-")[2].split("x")[0]


# In[ ]:


def setup_dirs_logs(args):
    log_dir = './.logs/%s'%args.output_dirname
    writer = SummaryWriter(log_dir=log_dir)
    write_to_config_file(args.__dict__, log_dir)

    return writer


# In[ ]:


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


# In[2]:


def bin_direction(direction):
    """takes 2D unit vector of direction and bins it to 4 directions right,down,left,up
       0: right
       1: down
       2: left
       3: up """
    
    minigrid_directions = [[1, 0],[ 0, -1],[-1 , 0],[0, 1]]
    binned_direction = minigrid_directions.index(list(direction))
    return binned_direction


# In[3]:


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
    


# In[1]:


def classification_acc(logits,true):
    guess = torch.argmax(logits,dim=1)
    acc = (float(torch.sum(torch.eq(true,guess)).data) / true.size(0))
    return acc


# In[6]:



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
        


# In[7]:


def get_trans_tuple():
        tuple_fields = ['x0','x1','a', 'r', 'done']
        

        tuple_fields.extend(['x0_coord_x', 'x0_coord_y', 'x1_coord_x', 'x1_coord_y'])

        tuple_fields.extend(["x0_direction","x1_direction"])
        
        Transition = namedtuple("Transition",tuple(tuple_fields))
        return Transition
        


# In[8]:


def create_zip_all(grid_size=(6,6),num_directions=4, num_actions=3):
    all_coord_x,all_coord_y = range(1,grid_size[0] - 1), range(1,grid_size[1] - 1)
    all_directions = range(num_directions)
    all_actions = range(num_actions)
    
    return product(all_coord_x,all_coord_y,all_directions,all_actions)
    


# In[9]:


def collect_one_unique_data_point(old_coords,old_directions,old_actions):
    old_zip_all = zip(old_coords,old_directions,old_actions)
    return old_zip_all


# In[10]:


def collect_datapoint(x0, action, env, convert_fxn):
    x0 = convert_fxn(x0,to_tensor=False)
    
    
    Transition = get_trans_tuple()
    
    # to make the coords start at 0
    x0_coord_x, x0_coord_y  = int(env.agent_pos[0]) - 1, int(env.agent_pos[1]) -1
    x0_direction = bin_direction(env.get_dir_vec())
    
    
    _, reward, done, _ = env.step(action)
    x1 = convert_fxn(env.render("rgb_array"))
    trans_list =  [x0,x1,action,reward,done]
    

    x1_coord_x, x1_coord_y = int(env.agent_pos[0]), int(env.agent_pos[1])
    trans_list.extend([x0_coord_x, x0_coord_y,x1_coord_x, x1_coord_y])

    x1_direction = bin_direction(env.get_dir_vec())
    trans_list.extend([x0_direction, x1_direction])
    return Transition(*trans_list)
        
    
    


# In[11]:


def get_desired_direction(env,desired_direction):
    desired_direction_vec = unbin_direction(desired_direction)
    true_direction_vec = env.get_dir_vec()
    while not np.allclose(true_direction_vec,desired_direction_vec):
        #print(true_direction_vec,desired_direction_vec)
        #print(np.allclose(true_direction_vec,desired_direction_vec))
        # turn right?
        
        try:
            _ = env.step(0)
        except:
            print(env.agent_pos)
            print(env.get_dir_vec())
            
        true_direction_vec = env.get_dir_vec()
    return env
    

def collect_specific_datapoint(env, coords, direction, action, convert_fxn):
    env.agent_pos = np.asarray(coords)
    env = get_desired_direction(env,direction)
    x0 = env.render("rgb_array")
    trans_obj  = collect_datapoint(x0, action, env, convert_fxn)
    return trans_obj
    

def collect_one_data_point(env=gym.make("MiniGrid-Empty-6x6-v0"),
                     policy=lambda x0: np.random.choice(3),
                     convert_fxn=convert_frame):
    x0 = env.render("rgb_array")
    action = policy(convert_fxn(x0,to_tensor=True))
    transitions_obj = collect_datapoint(x0, action, env, convert_fxn)
    return transitions_obj


def rollout_iterator(env=gym.make("MiniGrid-Empty-6x6-v0"),
                     policy=lambda x0: np.random.choice(3),
                     convert_fxn=convert_frame):
    _ = env.reset()
    env.seed(np.random.randint(100))
    env.agent_pos = env.place_agent(size=(env.grid_size,env.grid_size ))
    done = False
    while not done:
        transition = collect_one_data_point(env, policy, convert_fxn)
        done = transition.done
        yield transition


# In[12]:


def get_list_from_buffers(other_buffer,*keys):
    Transition = other_buffer.Transition
    full_buffer_mem = other_buffer.memory
    one_big_trans = Transition(*zip(*full_buffer_mem))
    ret = [one_big_trans._asdict()[key] for key in keys ]
    return ret


# In[13]:


def unused_datapoints_iterator(other_buffer,env=gym.make("MiniGrid-Empty-16x16-v0"),
                               convert_fxn=convert_frame):

    all_zip = create_zip_all(grid_size=(env.grid_size, env.grid_size))
    
    used_coord_x,    used_coord_y,    used_directions,    used_actions = get_list_from_buffers(other_buffer,
                                        "x0_coord_x",
                                        "x0_coord_y",
                                        "x0_direction",
                                        "a")
    
    #return used_coords
    used_zip = zip(used_coord_x,used_coord_y,used_directions,used_actions)

    used_set = set(used_zip)

    all_set = set(all_zip)

    unused = all_set.difference(used_set)
    while len(unused) > 0:
        coord_x,coord_y, direction, action = unused.pop()
        transition = collect_specific_datapoint(env,(coord_x,coord_y),direction,action,convert_fxn=convert_fxn)
        yield transition


# In[14]:


def mkstr(key,args={}):
    d = args.__dict__
    return "=".join([key,str(d[key])])


# In[15]:


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


# In[16]:


def write_ims(index,rows,ims,name, iter_):
    num_ims = rows**2
    ims_grid = make_grid((ims.data[index] + 1) / 2, rows)
    writer.add_image(name, ims_grid, iter_)
    


# In[17]:


def write_to_config_file(dict_,log_dir):
    config_file_path = Path(log_dir) / "config.json"
    dict_string = json.dumps(dict_) + "\n"
    with open(config_file_path, "w") as f:
        f.write(dict_string)
    


# In[18]:


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


# In[20]:


def do_k_episodes(convert_fxn,env,policy,k=1,epsilon=0.1,):
    rewards = []
    with torch.no_grad():
        for ep in range(k):
            done = False
            env.reset()
            cum_reward = 0
            while not done:
                _,_,_,reward, done = collect_one_data_point(convert_fxn=convert_fxn,
                                                            env=env,
                                                            policy=policy)
                cum_reward += float(reward)
            rewards.append(cum_reward)
        return np.mean(rewards), rewards


# In[21]:


if __name__ == "__main__":
    from replay_buffer import create_and_fill_replay_buffer
    
    env = gym.make("MiniGrid-Empty-6x6-v0")
    rb = create_and_fill_replay_buffer(env=env,size=10, 
                                     other_buffers=[])

    used = set([(t.x0_coord_x,t.x0_coord_y,t.a,t.x0_direction) for t in rb.memory])

    unused = []
    for t in unused_datapoints_iterator(other_buffer=rb,env=env ):
        unused.append((t.x0_coord_x,t.x0_coord_y,t.a,t.x0_direction))
    unused_set = set(unused)

    assert used.isdisjoint(unused_set)
    assert len(used.union(unused_set)) == (env.grid_size - 2)**2 * 3 * 4


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


