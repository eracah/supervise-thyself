from collections import namedtuple
import torch
import numpy as np
import random
from functools import partial
from data.collectors import Transition, DataCollector
from functools import partial
from data.iterators import PolicyIterator  #, ListIterator, UnusedPointsIterator
import copy
from data.utils import setup_env, convert_frames,convert_frame
import math
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

        

            
    # def __add__(self,other_buffer):
    #     buf = copy.deepcopy(self)
    #     buf.memory = buf.memory + other_buffer.memory
    #     del other_buffer
    #     return buf
    
    def extend(self,list_of_other_buffers):
        for oth_buf in list_of_other_buffers:
            self.memory = self.memory + oth_buf.memory
        del list_of_other_buffers
            
    
   
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    
    def sample(self, batch_size=None, chunk_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        trans = random.sample(self.memory, batch_size)        
        return self._convert_raw_sample(trans)
    
    
            
    def _convert_raw_sample(self,transitions):
        """converts 8-bit RGB to float and pytorch tensor"""
        # puts all trans objects into one trans object
        trans = self._combine_transitions_into_one_big_one(transitions)
        batch = self._convert_fields_to_pytorch_tensors(trans)
        return batch
        
    def _combine_transitions_into_one_big_one(self,transitions):
        fields = []
        for i,field in enumerate(zip(*transitions)):
            if isinstance(field[0],list):
                new_field = np.stack([list_ for list_ in field])
                if str(new_field.dtype) == "bool":
                    new_field = new_field.astype("int")
                #print(field.shape,field)
            if isinstance(field[0],dict):
                new_field = {}
                for k in field[0].keys():
                    all_items_of_key_k = [dic[k] for dic in field]
                    array_of_items_of_key_k = np.stack([list_ for list_ in all_items_of_key_k])
                    new_field[k] = array_of_items_of_key_k

            fields.append(new_field)
        return Transition(*fields)
    
    def _convert_fields_to_pytorch_tensors(self,trans):
        tb_dict = trans._asdict()
        for k,v  in trans.state_param_dict.items():
            tb_dict["state_param_dict"][k] = torch.tensor(v).to(self.DEVICE)
        tb_dict["xs"] = torch.stack([convert_frames(np.asarray(trans.xs[i]),to_tensor=True,resize_to=(-1,-1)) for
                                                     i in range(len(trans.xs))]).to(self.DEVICE)
        tb_dict["actions"] = torch.from_numpy(np.asarray(trans.actions)).to(self.DEVICE)
        tb_dict["rewards"] = torch.from_numpy(np.asarray(trans.rewards)).to(self.DEVICE)
        tb_dict["dones"] = torch.tensor(trans.dones)
        batch = Transition(*list(tb_dict.values()))
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

class BufferFiller(object):
    """creates and fills replay buffers with transitions"""
    def __init__(self,args,policy=None, capacity=10000):
        #self.env = env
        self.args = args

            

        self.capacity = capacity

        

    def make_empty_buffer(self):
        return ReplayMemory(capacity=self.capacity, batch_size=self.args.batch_size)
    
    def fill(self,size, frames_per_trans=2):
        """fill with transitions by just following a policy"""
        buffer = self.make_empty_buffer()
        iterator = PolicyIterator(args=self.args)
        buffer = self.fill_using_iterator(buffer,size,iterator)
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
                if global_size % 100 == 0:
                    print(global_size)
            iterator.reset()
        return buffer

def worker_fill(size,args,index):
    print("worker %i beginning fill!"%index)
    bf = BufferFiller(args)
    buf = bf.fill(size)
    return buf
    
def multicore_fill(size,args):
    from multiprocessing import Pool
    p = Pool(args.workers)
    size_per_process = math.ceil(size / args.workers)
    wf = partial(worker_fill, size_per_process, args)
    bufs = p.map(wf,range(args.workers))
    buf1 = bufs[0]
    buf1.extend(bufs[1:])
    return buf1
#     kwargs = {k:v for k,v in args.__dict__.items() if k != "processes" and k!="num_frames"}
    
#     num_rollouts = math.ceil(args.num_frames / args.rollout_size)
#     num_rollouts_per_process = math.ceil(num_rollouts / args.processes)
#     kwargs["num_rollouts"] = num_rollouts_per_process
#     f = partial(save_dataset,**kwargs)
#     p.map(f,range(0,num_rollouts,num_rollouts_per_process))
    

    
if __name__ is "__main__":
    env, action_space, grid_size,\
    num_directions, tot_examples, random_policy = setup_env("MiniGrid-Empty-8x8-v0")

    bf = BufferFiller(env=env,policy=random_policy,)

    rp = bf.fill(128,frames_per_trans=2)

    for t in rp:
        print(t.state_param_dict["y_coord"].shape)
        
    t = rp.sample(20)
    print(t.state_param_dict["x_coord"].shape)