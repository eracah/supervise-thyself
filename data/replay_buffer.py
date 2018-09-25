from collections import namedtuple
import torch
import numpy as np
from utils import convert_frames,convert_frame
import random
import data.custom_grids
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from functools import partial
from data.collectors import get_trans_tuple, DataCollector
from functools import partial
from data.iterators import PolicyIterator, ListIterator, UnusedPointsIterator
import copy

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
        buf = copy.deepcopy(self)
        buf.memory = buf.memory + other_buffer.memory
        return buf
    
   
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    
    def sample(self, batch_size=None, chunk_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        trans = random.sample(self.memory, batch_size)        
        return self._convert_raw_sample(trans)
        
    
    def _convert_raw_sample(self,transitions):
        """converts 8-bit RGB to float and pytorch tensor"""
        # puts all trans objects into one trans object
        #trans_batch = self.Transition(*zip(*transitions))
        list_of_fields = [list(zip(*field)) for field in zip(*transitions)]
        trans = self.Transition(*list_of_fields)
        tb_dict = trans._asdict()
        #return tb_dict,trans

        tb_dict["x_coords"] = torch.tensor(trans.x_coords).long().to(self.DEVICE)
        tb_dict["y_coords"] = torch.tensor(trans.y_coords).long().to(self.DEVICE)
        tb_dict["directions"] = torch.tensor(trans.directions).long().to(self.DEVICE)
        tb_dict["xs"] = torch.stack([convert_frames(np.asarray(trans.xs[i]),to_tensor=True,resize_to=(-1,-1)) for
                                                     i in range(len(trans.xs))]).to(self.DEVICE)
        tb_dict["actions"] = torch.from_numpy(np.asarray(trans.actions)).to(self.DEVICE)
        tb_dict["rewards"] = torch.from_numpy(np.asarray(trans.rewards)).to(self.DEVICE)
        tb_dict["dones"] = torch.tensor(trans.dones)
        batch = self.Transition(*list(tb_dict.values()))
        return batch

    def get_zipped_list(self,keys=["x_coords",
                                                "y_coords",
                                                "directions"], unique=False, concatenate=True):


        one_big_trans = self.Transition(*zip(*self.memory))
        #grab each field and concatenate all lists
        separated_fields = [one_big_trans._asdict()[key] for key in keys ]
        if concatenate:
            separated_fields = [np.concatenate(field) for field in separated_fields]
        ret_list = list(zip(*separated_fields))
        if unique:
#             if not concatenate:
#                 #concatenate each transition into one list and take set of it
#                 unique_ret_list = list(set([tuple(np.concatenate(r)) for r in ret_list]))
#                 # resplit back into list
#                 ret_list = [[list(a) for a in np.array_split(np.asarray(url),len(keys))] for url in unique_ret_list]
#             else:
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
        buf1 = buffer
#         unique_coords = copy.deepcopy(buffer.get_zipped_list(unique=True,concatenate=False))
#         len_coords = len(unique_coords)
        len_mem = len(buf1.memory)
        split_ind = int(proportion*len_mem)
        random.shuffle(buf1.memory)
        buf2 = copy.deepcopy(buf1)
        buf2.memory[:split_ind] = []
        buf1.memory[split_ind:] = []
        return buf1, buf2
        
        
    def make_empty_buffer(self):
        return ReplayMemory(capacity=self.capacity, batch_size=self.batch_size)
    
    def fill(self,size, frames_per_trans=2):
        """fill with transitions by just following a policy"""
        buffer = self.make_empty_buffer()
        iterator = PolicyIterator(policy=self.policy,
                                  env=self.env, 
                                  convert_fxn=self.convert_fxn,
                                  frames_per_trans=frames_per_trans)
        buffer = self.fill_using_iterator(buffer,size,iterator)
        return buffer  

    def fill_with_unvisited_states(self, visited_buffer,size):
        """fill with transitions not present in 'visited_buffer' """
        buffer = self.make_empty_buffer()
        visited_list = copy.deepcopy(visited_buffer.get_zipped_list(concatenate=True))
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