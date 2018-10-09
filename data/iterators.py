import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from utils import convert_frame
from data.collectors import get_trans_tuple
import numpy as np
import random
from itertools import product
from data.collectors import DataCollector
from utils import convert_frame
from utils import setup_env

def create_zip_all(grid_size=(6,6),num_directions=4):
    all_coord_x,all_coord_y = range(1,grid_size[0] - 1), range(1,grid_size[1] - 1)
    all_directions = range(num_directions)
    
    return product(all_coord_x,all_coord_y,all_directions)



class BaseIterator(object):
    """base iterator"""
    def __init__(self, env=gym.make("MiniGrid-Empty-6x6-v0"),
                     convert_fxn=convert_frame):
        self.convert_fxn = convert_fxn
        self.env = env
        
    def __iter__(self):
        return self
    
    def __next__(self):
        return self._next()
    
    def _next(self):
        raise NotImplementedError()
        
    def reset(self):
        return
        
    
    

class PolicyIterator(BaseIterator):
    """iterates datapoints following a policy"""
    def __init__(self, policy,
                        env,
                     convert_fxn=convert_frame,frames_per_trans=2,stop_at_done=True ):
        super(PolicyIterator,self).__init__(env,convert_fxn)
        self.dc = DataCollector(policy=policy,env=env,convert_fxn =convert_fxn,frames_per_trans=frames_per_trans)
        self.policy = policy
        self.done = False
        self.stop_at_done = stop_at_done
        self.reset()
        

    def reset(self):
        _ = self.env.reset()
        #self.env.agent_pos = self.env.place_agent(size=(self.env.grid_size,self.env.grid_size ))
        self.done = False
        
    def _next(self):
        if self.done:
            if self.stop_at_done:
                raise StopIteration()
            else:
                self.reset()
                
        transition = self.dc.collect_transition_per_the_policy()
        self.done = True in transition.dones
        return transition
    

if __name__ == "__main__":
    env, action_space, grid_size,\
    num_directions, tot_examples, random_policy = setup_env("originalGame-v0")
    pi = PolicyIterator(policy=random_policy, env=env, frames_per_trans=2)
    from matplotlib import pyplot as plt

    #%matplotlib inline


    for i, trans in enumerate(pi):
        print(i,env.env.game_state.game.numactions)