
# coding: utf-8

# In[4]:


import random
from collections import namedtuple
import torch
import numpy as np
from utils import convert_frames,convert_frame, rollout_iterator, plot_test
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from functools import partial


# In[21]:




class ReplayMemory(object):
    """Memory is uint8 to save space, then when you sample it converts to float tensor"""
    def __init__(self, capacity=10**6, batch_size=64, with_agent_pos=False):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = []
        self.position = 0
        self.with_agent_pos = with_agent_pos
        if self.with_agent_pos:
            self.Transition = namedtuple('Transition',
                        ('state','next_state','action', 'reward', 'done',
                         'state_coords', 'next_state_coords'))
        else:
            self.Transition = namedtuple('Transition',
                        ('state','next_state','action', 'reward', 'done'))

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        if self.with_agent_pos:
            x0s, x1s,as_, rs, dones,x0s_coords,x1s_coords = zip(*transitions)
            x0s_coords = torch.tensor(x0s_coords).long().to(self.DEVICE)
            x1s_coords = torch.tensor(x1s_coords).long().to(self.DEVICE)
            
        else:  
            x0s, x1s,as_, rs, dones = zip(*transitions)
        x0, x1 = convert_frames(np.asarray(x0s),to_tensor=True,resize_to=(-1,-1)),                convert_frames(np.asarray(x1s),to_tensor=True,resize_to=(-1,-1))
        a,r = torch.from_numpy(np.asarray(as_)), torch.from_numpy(np.asarray(rs)),
        x0,x1,a,r = x0.to(self.DEVICE),x1.to(self.DEVICE),a.to(self.DEVICE),r.float().to(self.DEVICE)
        batch = [x0,x1,a,r, dones]
        if self.with_agent_pos:
            batch.extend([x0s_coords,x1s_coords])
        return batch
        
    def __iter__(self):
        while True:
            yield self.sample(self.batch_size)
    def __len__(self):
        return len(self.memory)


# In[22]:


def fill_replay_buffer(buffer,
                       size,
                       rollout_size=256,
                       env = gym.make("MiniGrid-Empty-6x6-v0"),
                       resize_to = (64,64),
                       policy= lambda x0: np.random.choice(3),
                       with_agent_pos=False
                      ):
    #fills replay buffer with size examples
    convert_fxn = partial(convert_frame,resize_to=resize_to)
    num_rollouts = int(np.ceil(size / rollout_size))
    global_size=0
    for rollout in range(num_rollouts):
        for i, transition in enumerate(rollout_iterator(env=env,
                                                        convert_fxn=convert_fxn,
                                                        policy=policy,
                                                       get_agent_pos=with_agent_pos)):
            if global_size >= size:
                return
            buffer.push(*transition)
            global_size += 1

    

    
 


# In[23]:


if __name__ == "__main__":


    rm  = ReplayMemory(batch_size=10,capacity=20,with_agent_pos=True)

    fill_replay_buffer(rm,21,with_agent_pos=True)

    x0,x1,a,r,done,x0_coords,x1_coords = rm.sample(batch_size=10)

    plot_test(x0,x1,a,r, label_list=["left","right","forward"] )

