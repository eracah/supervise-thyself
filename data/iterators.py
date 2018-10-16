from data.collectors import Transition
import numpy as np
import random
from itertools import product
from data.collectors import DataCollector
from data.utils import convert_frame, setup_env

def create_zip_all(grid_size=(6,6),num_directions=4):
    all_coord_x,all_coord_y = range(1,grid_size[0] - 1), range(1,grid_size[1] - 1)
    all_directions = range(num_directions)
    
    return product(all_coord_x,all_coord_y,all_directions)



class BaseIterator(object):
    """base iterator"""
    def __init__(self):
        pass
        
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
    def __init__(self,args,policy=None, stop_at_done=True ):
        super(PolicyIterator,self).__init__()
        self.dc = DataCollector(args=args, policy=policy)
        self.done = False
        self.stop_at_done = stop_at_done
        self.reset()
        

    def reset(self):
        _ = self.dc.env.reset()
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