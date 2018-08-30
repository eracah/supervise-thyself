
# coding: utf-8

# In[1]:


import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from utils import convert_frame, bin_direction, unbin_direction, get_trans_tuple
import numpy as np
import random
from itertools import product
from collectors import DataCollector


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


# In[7]:


############ TESTS ##############


# In[8]:


def test_unused_points_iterator():
    env = gym.make("MiniGrid-Empty-6x6-v0")
    num_dirs = 4
    num_actions = 3
    grid_list = range(1,env.grid_size - 1)
    dir_list = range(num_dirs)
    act_list = range(num_actions)
    ch = np.random.choice
    size = 1000
    x = ch(grid_list,size=size)
    y = ch(grid_list,size=size)
    d = ch(dir_list,size=size)
    a = ch(act_list,size=size)

    used = list(zip(x,y,d,a))
    ui = UnusedPointsIterator(used)

    unused = ui.get_unused_datapoints(used, env)
    unused = []
    for t in ui:
        unused.append((t.x0_coord_x,t.x0_coord_y,t.x0_direction,t.a))
    unused_set = set(unused)
    used_set = set(used)
    assert used_set.isdisjoint(set(unused_set))
    assert len(used_set.union(unused_set)) == (env.grid_size - 2)**2 * 3 * 4
    


# In[9]:


def test_policy_iterator():
    
    pi = PolicyIterator()
    
    # test continuing where you left off
    last_step = 0
    for i,g in enumerate(pi):
        last_step = pi.env.step_count

        if i == 5:
            break

    for i,g in enumerate(pi):
        if i == 0:
            assert pi.env.step_count == last_step + 1

    # test full reset
    pi.reset()
    for i,g in enumerate(pi):
        if i == 0:
            assert pi.env.step_count == 1


# In[10]:


def test_list_iterator():
    grid_list = range(1,5)
    dir_list = range(4)
    act_list = range(3)
    ch = np.random.choice
    size =100
    x = ch(grid_list,size=size)
    y = ch(grid_list,size=size)
    d = ch(dir_list,size=size)
    a = ch(act_list,size=size)

    list_of_points = list(zip(x,y,d,a))
    #print(list_of_points)
    list_it = ListIterator(list_of_points)
    test_list = []
    for i,t in enumerate(list_it):
        trans_tup = tuple([getattr(t,k) for k in ["x0_coord_x",
                                            "x0_coord_y",
                                            "x0_direction",
                                            "a"] ])

        test_list.append(trans_tup)
    assert i == len(list_of_points) - 1
    assert test_list == list_of_points


# In[11]:


if __name__ == "__main__":
    test_list_iterator()
    test_policy_iterator()
    test_unused_points_iterator()

