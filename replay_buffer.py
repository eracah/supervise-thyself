
# coding: utf-8

# In[1]:


import random
from collections import namedtuple
import torch
import numpy as np
from load_data import convert_frames,convert_frame, DataCreator, plot_test


# In[6]:


Transition = namedtuple('Transition',
                        ('state','next_state','action', 'reward', 'done'))


class ReplayMemory(object):
    """Memory is uint8 to save space, then when you sample it converts to float tensor"""
    def __init__(self, capacity=10**6, batch_size=64):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = []
        self.position = 0

    def push(self, **kwargs):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(**kwargs)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        x0s, x1s,as_, rs, dones = zip(*transitions)
        x0, x1 = convert_frames(np.asarray(x0s),to_tensor=True,resize_to=(-1,-1)), convert_frames(np.asarray(x1s),to_tensor=True,resize_to=(-1,-1))
        a,r = torch.from_numpy(np.asarray(as_)), torch.from_numpy(np.asarray(rs)),
        x0,x1,a,r = x0.to(self.DEVICE),x1.to(self.DEVICE),a.to(self.DEVICE),r.float().to(self.DEVICE)
        return x0,x1,a,r, dones
        
    def __iter__(self):
        while True:
            yield self.sample(self.batch_size)
    def __len__(self):
        return len(self.memory)


# In[7]:


# if __name__ == "__main__":
#     rm = ReplayMemory(100,5)

#     dc = DataCreator()
#     x0,x1,a, r = dc.do_rollout()

#     for i in range(x0.shape[0]):
#         rm.push(state=x0[i],next_state=x1[i],action=a[i], reward=r[i])


# In[8]:


def fill_replay_buffer(buffer,size, rollout_size=256,
                       env_name="MiniGrid-Empty-6x6-v0",
                       resize_to = (64,64),
                       action_space = range(3)):
    #fills replay buffer with size examples
    num_rollouts = int(np.ceil(size / rollout_size))
    dc = DataCreator(env_name=env_name,
                     resize_to = resize_to,
                     action_space=action_space,
                     rollout_size=rollout_size)
    for rollout in range(num_rollouts):
        for i, (x0,x1,a,r,done) in enumerate(dc.rollout_iterator()):
            buffer.push(state=x0,action=a,next_state=x1,reward=r,done=done)
    

    
 


# In[9]:


if __name__ == "__main__":


    rm  = ReplayMemory(batch_size=10,capacity=20)

    fill_replay_buffer(rm,21)

    x0,x1,a,r,done = rm.sample(batch_size=10)

    plot_test(x0,x1,a,r, label_list=["left","right","forward"] )

