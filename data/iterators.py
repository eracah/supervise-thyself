
# coding: utf-8

# In[1]:


import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from utils import convert_frame, bin_direction, unbin_direction, get_trans_tuple
import numpy as np
import random
from itertools import product
from data.collectors import DataCollector


# In[2]:


def create_zip_all(grid_size=(6,6),num_directions=4, num_actions=3):
    all_coord_x,all_coord_y = range(1,grid_size[0] - 1), range(1,grid_size[1] - 1)
    all_directions = range(num_directions)
    all_actions = range(num_actions)
    
    return product(all_coord_x,all_coord_y,all_directions,all_actions)


# In[3]:


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
        
    
    


# In[4]:


class PolicyIterator(BaseIterator):
    """iterates datapoints following a policy"""
    def __init__(self, policy=lambda x0: np.random.choice(3),
                        env=gym.make("MiniGrid-Empty-6x6-v0"),
                     convert_fxn=convert_frame):
        super(PolicyIterator,self).__init__(env,convert_fxn)
        self.dc = DataCollector(policy=policy,env=env,convert_fxn =convert_fxn)
        self.policy = policy
        self.done = False
        self.reset()
        

    def reset(self):
        _ = self.env.reset()
        self.env.seed(np.random.randint(100))
        self.env.agent_pos = self.env.place_agent(size=(self.env.grid_size,self.env.grid_size ))
        self.done = False
        
    def _next(self):
        if not self.done:
            transition = self.dc.collect_data_point_per_policy()
            self.done = transition.done
            return transition
        else:
            raise StopIteration()
    
    
    
    


# In[5]:


class ListIterator(BaseIterator):
    """takes a list of tuples (coord,direction,action) and 
    renders that to full s,a,s transitions"""
    def __init__(self, list_of_points,env=gym.make("MiniGrid-Empty-6x6-v0"),
                     convert_fxn=convert_frame):
        super(ListIterator,self).__init__(env,convert_fxn)
        self.dc = DataCollector(env=env,convert_fxn =convert_fxn)
        self.list_of_points = list_of_points
        self.i = 0

    def __len__(self):
        return len(self.list_of_points)
    
    def _next(self):
        if self.i < len(self.list_of_points):
            (coord_x,coord_y, direction, action) = self.list_of_points[self.i]
            transition = self.dc.collect_specific_datapoint((coord_x ,coord_y),
                                                    direction,
                                                    action)

            self.i += 1
            return transition
        else:
            raise StopIteration()
        
        


# In[6]:


class UnusedPointsIterator(ListIterator):
    """takes a s,a,s list and iterates all state,action,state
    triplets not in the buffer"""
    def __init__(self,used_list, 
                 env=gym.make("MiniGrid-Empty-6x6-v0"),
                     convert_fxn=convert_frame):
        
        unused_list = self.get_unused_datapoints(used_list,env)
        
        super(UnusedPointsIterator, self).__init__(unused_list,env, convert_fxn)
    
    def get_unused_datapoints(self, used_list, env):
        all_zip = create_zip_all(grid_size=(env.grid_size, env.grid_size))
        all_set = set(all_zip)
        used_set = set(used_list)
        unused = all_set.difference(used_set)
        
        assert unused.isdisjoint(used_set)

        return list(unused)

