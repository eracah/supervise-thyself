
# coding: utf-8

# In[8]:


import data.custom_grids
import random
from collections import namedtuple
import torch
import numpy as np
from utils import convert_frames,convert_frame
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from functools import partial
from utils import get_trans_tuple
from functools import partial
from data.iterators import PolicyIterator, ListIterator, UnusedPointsIterator
#from torch.utils.data import 
import copy


# In[1]:


class ReplayMemory(object):
    """buffer of transitions. you can sample it like a true replay buffer (with replacement) using self.sample
    or like normal data iterator used in most supervised learning problems with sellf.__iter__()"""
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

    def get_zipped_list(self,keys=["x0_coord_x",
                                                "x0_coord_y",
                                                "x0_direction",
                                                "a"], unique=False):


    
        one_big_trans = self.Transition(*zip(*self.memory))
        separated_fields = [one_big_trans._asdict()[key] for key in keys ]
        ret_list = list(zip(*separated_fields))
        if unique:
            ret_list = list(set(ret_list))
        return ret_list
    
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


# In[ ]:


class BufferFiller(object):
    """creates and fills replay buffers with transitions"""
    def __init__(self,
                 convert_fxn = partial(convert_frame, resize_to=(64,64)),
                 env = gym.make("MiniGrid-Empty-8x8-v0"),
                 policy= lambda x0: np.random.choice(3), capacity=10000,batch_size=32):
        self.env = env
        self.policy = policy
        self.convert_fxn = convert_fxn
        self.capacity = capacity
        self.batch_size = batch_size
        

    def split(self, buffer, proportion):
        unique_coords = copy.deepcopy(list(set(buffer.get_zipped_list(unique=True))))
        len_coords = len(unique_coords)
        split_ind = int(proportion*len_coords)
        tr_list = unique_coords[:split_ind]

        val_list = unique_coords[split_ind:]
        buf1 = self.fill_with_list(tr_list)
        buf2 = self.fill_with_list(val_list)
        del buffer
        return buf1, buf2
        
        
    def make_empty_buffer(self):
        return ReplayMemory(capacity=self.capacity, batch_size=self.batch_size)
    
    def fill(self,size):
        """fill with transitions by just following a policy"""
        buffer = self.make_empty_buffer()
        iterator = PolicyIterator(policy=self.policy,
                                  env=self.env, 
                                  convert_fxn=self.convert_fxn)
        buffer = self.fill_using_iterator(buffer,size,iterator)
        return buffer  

    def fill_with_unvisited_states(self, visited_buffer,size):
        """fill with transitions not present in 'visited_buffer' """
        buffer = self.make_empty_buffer()
        visited_list = copy.deepcopy(visited_buffer.get_zipped_list())
        iterator = UnusedPointsIterator(visited_list, 
                                        env=self.env,
                                        convert_fxn=self.convert_fxn)
        if size > len(iterator) or size == -1:
            size = len(iterator)
        assert len(iterator) > 0
        buffer = self.fill_using_iterator(buffer,size,iterator)
        
        assert set(visited_buffer.get_zipped_list()).isdisjoint(set(buffer.get_zipped_list()))
        return buffer
    
    def fill_with_list(self,list_, size=-1):
        """fill with transitions specified in a list of coordinates and actions """
        buffer = self.make_empty_buffer()
        iterator = ListIterator(list_of_points=list_,
                                env=self.env,
                                convert_fxn=self.convert_fxn)
        
        if size > len(iterator) or size == -1:
            size = len(iterator)

        buffer = self.fill_using_iterator(buffer,size, iterator)
        return buffer
    
    def fill_using_iterator(self, buffer,size, iterator):
        """fill using the given transition iterator """
        
        global_size=0
        while global_size < size:
            for i, transition in enumerate(iterator):
                buffer.push(*transition)
                global_size += 1
                if global_size >= size:
                    return buffer
            iterator.reset()
        return buffer


