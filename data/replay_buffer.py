from collections import namedtuple
import torch
import numpy as np
import random
from functools import partial
from data.collectors import EpisodeCollector
from functools import partial
import copy
from data.utils import setup_env, convert_frames,convert_frame
import math

class ReplayMemory(object):
    """buffer of transitions. you can sample it like a true replay buffer (with replacement) using self.sample
    or like normal data iterator used in most supervised learning problems with sellf.__iter__()"""
    """Memory is uint8 to save space, then when you sample it converts to float tensor"""
    def __init__(self,args, batch_size=64):
        self.args = args
        self.DEVICE = self.args.device
        self.batch_size = batch_size
        self.episodes = []                
   
    def push(self, episode_trans):
        """Saves a transition."""
        self.episodes.append(episode_trans)

    
    def sample(self, batch_size=None, chunk_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        trans = random.sample(self.memory, batch_size)        
        return self._convert_raw_sample(trans)
    
    def sample_func(self,ep_ind,frame_ind, num):
        raise NotImplementedError
    
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
    def __init__(self,args,policy=None):
        #self.env = env
        self.args = args
        self.policy=policy

    def make_empty_buffer(self):
        return ReplayMemory(batch_size=self.args.batch_size, args=self.args)
    
    def fill(self,size):
        """fill with transitions by just following a policy"""
        buffer = self.make_empty_buffer()
        collector = EpisodeCollector(args=self.args,policy=self.policy)
        buffer = self._fill(size, collector, buffer)
        return buffer 
    
    def _fill(self,size, collector, buffer):
        cur_size = 0
        while cur_size < size:
            episode = collector.collect_episode_per_the_policy()
            cur_size += len(episode.xs)
            buffer.push(episode)
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
    