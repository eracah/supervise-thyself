
# coding: utf-8

# In[1]:


import custom_grids
import random
from collections import namedtuple
import torch
import numpy as np
from utils import setup_env,convert_frames,convert_frame, rollout_iterator, plot_test,unused_datapoints_iterator
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from functools import partial
from utils import get_trans_tuple
from functools import partial
#from torch.utils.data import 
import copy


# In[12]:


class ReplayMemory(object):
    """Memory is uint8 to save space, then when you sample it converts to float tensor"""
    def __init__(self, capacity=10**6, batch_size=64):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = []
        self.position = 0

        
        self.Transition = get_trans_tuple()
            

    def __add__(self,other_buffer):
        self.memory = self.memory + other_buffer.memory
        return self
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    
    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        trans = random.sample(self.memory, batch_size)
        return self._convert_raw_sample(trans)
        
    
    def _convert_raw_sample(self,transitions):
        trans_batch = self.Transition(*zip(*transitions))
        tb_dict = trans_batch._asdict()

        tb_dict["x0_coord_x"] = torch.tensor(trans_batch.x0_coord_x).long().to(self.DEVICE)
        tb_dict["x0_coord_y"] = torch.tensor(trans_batch.x0_coord_y).long().to(self.DEVICE)
        tb_dict["x1_coord_x"] = torch.tensor(trans_batch.x1_coord_x).long().to(self.DEVICE)
        tb_dict["x1_coord_y"] = torch.tensor(trans_batch.x1_coord_y).long().to(self.DEVICE)

        tb_dict["x0_direction"] = torch.tensor(trans_batch.x0_direction).long().to(self.DEVICE)
        tb_dict["x1_direction"] = torch.tensor(trans_batch.x1_direction).long().to(self.DEVICE)


            
        tb_dict["x0"] = convert_frames(np.asarray(trans_batch.x0),to_tensor=True,resize_to=(-1,-1)).to(self.DEVICE)
        tb_dict["x1"] = convert_frames(np.asarray(trans_batch.x1),to_tensor=True,resize_to=(-1,-1)).to(self.DEVICE)
        tb_dict["a"] = torch.from_numpy(np.asarray(trans_batch.a)).to(self.DEVICE)
        tb_dict["r"] = torch.from_numpy(np.asarray(trans_batch.r)).to(self.DEVICE)
        
        batch = self.Transition(*list(tb_dict.values()))
        return batch
        
    def __iter__(self):
        """Iterator that samples without replacement for replay buffer
        It's basically like a standard sgd setup
        If you want to sample with replacement like standard replay buffer use self.sample"""
        mem = copy.deepcopy(self.memory)
        random.shuffle(mem)
        size = len(mem)
        for st in range(0, size, self.batch_size):
            end = st+self.batch_size if st+self.batch_size <= size else size
            raw_sample = self.memory[st:end]
            yield self._convert_raw_sample(raw_sample)
    
    def __len__(self):
        return len(self.memory)


# In[13]:


class BufferFiller(object):
    def __init__(self,
                 convert_fxn = partial(convert_frame, resize_to=(64,64)),
                 env = gym.make("MiniGrid-Empty-16x16-v0"),
                 policy= lambda x0: np.random.choice(3)):
        self.env = env
        self.policy = policy
        self.convert_fxn = convert_fxn
        


    def create_and_fill(self,size=1000,
                        batch_size=32,
                        capacity=10000, 
                        conflicting_buffer=None):

        buffer = ReplayMemory(capacity=capacity,
                                   batch_size=batch_size)

        if not conflicting_buffer:
            buffer = self.fill_buffer_with_rollouts(buffer,size)
        else:
            buffer = self.fill_buffer_with_unique_transitions(conflicting_buffer,buffer,size)

        return buffer
    
    def fill_buffer_with_rollouts(self,buffer,size):
        iterator = partial(rollout_iterator,env=self.env,
                                    convert_fxn=self.convert_fxn,
                                    policy=self.policy)
        buffer = self.fill_buffer_with_iterator(buffer,size,iterator)
        return buffer  

    def fill_buffer_with_unique_transitions(self,other_buffer,buffer,size):
        iterator = partial(unused_datapoints_iterator, other_buffer=other_buffer,env=self.env,
                                              convert_fxn=self.convert_fxn)
        buffer= self.fill_buffer_with_iterator(buffer,size,iterator)
        return buffer
    
    def fill_buffer_with_iterator(self,buffer,size, iterator):
        global_size=0
        while True:
            for i, transition in enumerate(iterator()):
                buffer.push(*transition)
                global_size += 1
                if global_size >= size:
                    return buffer




        

    

    
 


# In[15]:


if __name__ == "__main__":
    bf = BufferFiller()
    
    rb = bf.create_and_fill(size=100)

    val_rb = bf.create_and_fill(size=10,conflicting_buffer=rb)
    
    tst_rb = bf.create_and_fill(size=10,conflicting_buffer=rb+val_rb)
    vt = val_rb.sample(5)

    vts = set([vt.x0_coord_x,vt.x0_coord_y,vt.a,vt.x0_direction])

    rt = rb.sample(15)

    rts = set([rt.x0_coord_x,rt.x0_coord_y,rt.a,rt.x0_direction])

    tt = tst_rb.sample(5)
    tts = set([tt.x0_coord_x,tt.x0_coord_y,tt.a,tt.x0_direction])
    assert rts.isdisjoint(vts)
    assert rts.isdisjoint(tts)
    assert vts.isdisjoint(tts)

