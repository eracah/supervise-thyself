
# coding: utf-8

# In[1]:


from utils import get_trans_tuple, convert_frame, bin_direction, unbin_direction
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
import numpy as np


# In[6]:


class DataCollector(object):
    def __init__(self, policy=lambda x0: np.random.choice(3),
                        env=gym.make("MiniGrid-Empty-6x6-v0"),
                     convert_fxn=convert_frame):
        self.convert_fxn = convert_fxn
        self.policy = policy
        self.env = env

    def _collect_datapoint(self,x0, action):
        x0 = self.convert_fxn(x0,to_tensor=False)


        Transition = get_trans_tuple()

        # to make the coords start at 0
        x0_coord_x, x0_coord_y  = int(self.env.agent_pos[0]), int(self.env.agent_pos[1])
        x0_direction = bin_direction(self.env.get_dir_vec())


        _, reward, done, _ = self.env.step(action)
        x1 = self.convert_fxn(self.env.render("rgb_array"))
        trans_list =  [x0,x1,action,reward,done]

        # to make the coords start at 0
        x1_coord_x, x1_coord_y = int(self.env.agent_pos[0]), int(self.env.agent_pos[1])
        trans_list.extend([x0_coord_x, x0_coord_y,x1_coord_x, x1_coord_y])

        x1_direction = bin_direction(self.env.get_dir_vec())
        trans_list.extend([x0_direction, x1_direction])
        return Transition(*trans_list)

    def collect_data_point_per_policy(self):
        x0 = self.env.render("rgb_array")
        # to_tensor true just in case policy is exactly a neural network
        action = self.policy(self.convert_fxn(x0,to_tensor=True))
        transitions_obj = self._collect_datapoint(x0, action)
        return transitions_obj
    
    def collect_specific_datapoint(self,coords, direction, action):
        self.env.agent_pos = np.asarray(coords)
        self._get_desired_direction(direction)
        x0 = self.env.render("rgb_array")
        trans_obj  = self._collect_datapoint(x0, action)
        return trans_obj
    def _get_desired_direction(self,desired_direction):
        desired_direction_vec = unbin_direction(desired_direction)
        true_direction_vec = self.env.get_dir_vec()
        while not np.allclose(true_direction_vec,desired_direction_vec):
            _ = self.env.step(0)
            true_direction_vec = self.env.get_dir_vec()


# In[9]:


def test_collect_specific_datapoint():
    dc = DataCollector()
    env = gym.make("MiniGrid-Empty-6x6-v0")
    grid_list = range(1,5)
    dir_list = range(4)
    act_list = range(3)
    ch = np.random.choice

    for i in range(100):
        x = ch(grid_list)
        y = ch(grid_list)
        d = ch(dir_list)
        a = ch(act_list)
        inp_tup = (x,y,d,a)
        trans = dc.collect_specific_datapoint((x,y), d, a)
        out_tup = tuple([getattr(trans,k) for k in ["x0_coord_x",
                                                "x0_coord_y",
                                                "x0_direction",
                                                "a"]])
        assert inp_tup == out_tup, "%s,%s"%(str(inp_tup),str(out_tup))


# In[11]:


def test_collect_data_point_per_policy():
    dc = DataCollector()
    for i in range(100):
        trans = dc.collect_data_point_per_policy()


# In[12]:


if __name__ == "__main__":
    test_collect_data_point_per_policy()
    test_collect_specific_datapoint()

