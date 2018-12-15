from collections import namedtuple
import torch
import numpy as np
import random
from functools import partial
from data.samplers import FrameActionSampler, FrameSampler
import copy
from data.utils import setup_env, convert_frames,convert_frame
import math
from data.collectors import EpisodeCollector

class SamplerFiller(object):
    """creates and fills replay buffers with transitions"""
    def __init__(self, args, policy=None):
        #self.env = env
        self.args = args
        self.policy=policy
        self.Sampler = FrameActionSampler if self.args.there_are_actions else FrameSampler

    def make_empty_buffer(self):
        return self.Sampler(batch_size=self.args.batch_size, args=self.args)
    
    def fill(self,size):
        """fill with transitions by just following a policy"""
        buffer = self.make_empty_buffer()
        collector = EpisodeCollector(args=self.args,policy=self.policy)
        sampler = self._fill(size, collector, buffer)
        return sampler 
    
    def _fill(self,size, collector, sampler):
        cur_size = 0
        while cur_size < size:
            num_left = (size - cur_size) #*self.args.frames_per_example
            episode = collector.collect_episode_per_the_policy(max_frames=num_left)
            episode_len = len(episode.xs)
            cur_size += episode_len
            sampler.push(episode)
        return sampler
    
    
    
def worker_fill(size,args,index):
    print("worker %i beginning fill!"%index)
    sf = SamplerFiller(args)
    sampler = sf.fill(size)
    return sampler
    
def multicore_fill(size,args):
    from multiprocessing import Pool
    p = Pool(args.workers)
    size_per_process = math.ceil(size / args.workers)
    wf = partial(worker_fill, size_per_process, args)
    samplers = p.map(wf,range(args.workers))
    s1 = samplers[0]
    s1.extend(samplers[1:])
    return s1