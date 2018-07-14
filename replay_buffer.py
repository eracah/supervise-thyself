
# coding: utf-8

# In[1]:


import random
from collections import namedtuple
import torch
import numpy as np


# In[4]:


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity, batch_size):
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
        x0s, as_, x1s, rs = zip(*transitions)
        x0, a, x1, r = torch.stack(x0s), torch.cat(as_),                        torch.stack(x1s), torch.cat(rs)
        return x0,a,x1,r
        
    def __iter__(self):
        while True:
            yield self.sample(self.batch_size)
    def __len__(self):
        return len(self.memory)


# In[5]:




# rm = ReplayMemory(10,5)

# for i in range(10):
#     rm.push(*range(4))



# for i,tr in enumerate(rm):
#     print(i,tr)
#     rm.push(*range(i+4,i+8))
#     if i == 10:
#         break

