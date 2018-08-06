
# coding: utf-8

# In[7]:


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


# In[8]:


class ReplayMemory(object):
    """Memory is uint8 to save space, then when you sample it converts to float tensor"""
    def __init__(self, capacity=10**6, batch_size=64, **kwargs):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = []
        self.position = 0
        self.with_agent_pos = kwargs["with_agent_pos"] if "with_agent_pos" in kwargs else False
        self.with_agent_heading = kwargs["with_agent_heading"] if "with_agent_heading" in kwargs else False
        
        self.Transition = get_trans_tuple(self.with_agent_pos,
                                          self.with_agent_heading)
            

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

        if self.with_agent_heading:
            tb_dict["x0_heading"] = torch.tensor(trans_batch.x0_heading).long().to(self.DEVICE)
            tb_dict["x1_heading"] = torch.tensor(trans_batch.x1_heading).long().to(self.DEVICE)
            
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


# In[9]:


def fill_replay_buffer(buffer,
                       size,
                       resize_to = (64,64),
                       env = gym.make("MiniGrid-Empty-6x6-v0"),
                       policy= lambda x0: np.random.choice(3),
                       **kwargs
                      ):
    #fills replay buffer with size examples
    rollout_size = env.max_steps
    convert_fxn = partial(convert_frame, resize_to=resize_to)
    num_rollouts = int(np.ceil(size / rollout_size))
    global_size=0
    for rollout in range(num_rollouts):
        for i, transition in enumerate(rollout_iterator(env=env,
                                                        convert_fxn=convert_fxn,
                                                        policy=policy,
                                                       **kwargs)):
            if global_size >= size:
                return
            buffer.push(*transition)
            global_size += 1

    

    
 


# In[10]:


def setup_replay_buffer(capacity=10000, 
                        batch_size=8, 
                        init_buffer_size=128, 
                        env= gym.make("MiniGrid-Empty-6x6-v0"),
                        resize_to = (64,64),
                        action_space=np.arange(3), **kwargs):
    #print("setting up buffer")
    replay_buffer = ReplayMemory(capacity=capacity,
                                 batch_size=batch_size,**kwargs)
    
    fill_replay_buffer(buffer=replay_buffer,
                       size=init_buffer_size,
                       env = env,
                       resize_to=resize_to,
                       policy= lambda x0: np.random.choice(action_space),
                       **kwargs
                      )
    #print("buffer filled!")
    return replay_buffer


# In[11]:


if __name__ == "__main__":
    rb = setup_replay_buffer( with_agent_pos=True, with_agent_heading=True)

    batch = rb.sample()

    plot_test(batch.x0,batch.x1,batch.a,batch.r, label_list=["left","right","forward"] )

