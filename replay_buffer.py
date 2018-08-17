
# coding: utf-8

# In[3]:


import custom_grids
import random
from collections import namedtuple
import torch
import numpy as np
from utils import setup_env,convert_frames,convert_frame, rollout_iterator, plot_test
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from functools import partial
from utils import get_trans_tuple


# In[4]:


class ReplayMemory(object):
    """Memory is uint8 to save space, then when you sample it converts to float tensor"""
    def __init__(self, capacity=10**6, batch_size=64, **kwargs):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = []
        self.position = 0
        self.with_agent_pos = kwargs["with_agent_pos"] if "with_agent_pos" in kwargs else False
        self.with_agent_direction = kwargs["with_agent_direction"] if "with_agent_direction" in kwargs else False
        
        self.Transition = get_trans_tuple(self.with_agent_pos,
                                          self.with_agent_direction)
            

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        trans = random.sample(self.memory, batch_size)
        trans_batch = self.Transition(*zip(*trans))
        tb_dict = trans_batch._asdict()
        if self.with_agent_pos:
            tb_dict["x0_coords"] = torch.tensor(trans_batch.x0_coords).long().to(self.DEVICE)
            tb_dict["x1_coords"] = torch.tensor(trans_batch.x1_coords).long().to(self.DEVICE)

        if self.with_agent_direction:
            tb_dict["x0_direction"] = torch.tensor(trans_batch.x0_direction).long().to(self.DEVICE)
            tb_dict["x1_direction"] = torch.tensor(trans_batch.x1_direction).long().to(self.DEVICE)
            
        tb_dict["x0"] = convert_frames(np.asarray(trans_batch.x0),to_tensor=True,resize_to=(-1,-1)).to(self.DEVICE)
        tb_dict["x1"] = convert_frames(np.asarray(trans_batch.x1),to_tensor=True,resize_to=(-1,-1)).to(self.DEVICE)
        tb_dict["a"] = torch.from_numpy(np.asarray(trans_batch.a)).to(self.DEVICE)
        tb_dict["r"] = torch.from_numpy(np.asarray(trans_batch.r)).to(self.DEVICE)
        
        batch = self.Transition(*list(tb_dict.values()))
        return batch
        
    def __iter__(self):
        while True:
            yield self.sample(self.batch_size)
    def __len__(self):
        return len(self.memory)


# In[5]:


def fill_buffer_with_rollouts(buffer,size, env, convert_fxn, policy,**kwargs):
    global_size=0
    while True:
        for i, transition in enumerate(rollout_iterator(env=env,
                                                        convert_fxn=convert_fxn,
                                                        policy=policy,
                                                       **kwargs)):
            buffer.push(*transition)
            global_size += 1
            if global_size >= size:
                return buffer

    


# In[6]:


def fill_buffer_with_unique_transitions(other_buffers,buffer,size, env, convert_fxn, policy,**kwargs):
    print(other_buffers)
    return buffer


# In[7]:


def create_and_fill_replay_buffer(size=1000,
                                  capacity=10000, 
                                  batch_size=32,
                                   resize_to = (64,64),
                                   env = gym.make("MiniGrid-Empty-6x6-v0"),
                                   policy= lambda x0: np.random.choice(3),
                                other_buffers=None,
                                   **kwargs):
    
    buffer = ReplayMemory(capacity=capacity,
                               batch_size=batch_size,**kwargs)
    convert_fxn = partial(convert_frame, resize_to=resize_to)
    if not other_buffers:
        buffer = fill_buffer_with_rollouts(buffer,size, env, convert_fxn, policy,**kwargs)
    else:
        buffer = fill_buffer_with_unique_transitions(other_buffers,buffer,size, env, convert_fxn, policy,**kwargs)
    
    return buffer
        

    

    
 


# In[12]:


if __name__ == "__main__":
    rb = create_and_fill_replay_buffer(size=10, with_agent_pos=True, with_agent_direction=True, other_buffers=[])
#     batch = rb.sample()

#     plot_test(batch.x0,batch.x1,batch.a,batch.r, label_list=["left","right","forward"] )

    get_list_from_buffers(rb,
                                            "x0_coords",
                                            "x0_direction",
                                            "a")

