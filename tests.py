
# coding: utf-8

# In[7]:


from data.replay_buffer import BufferFiller
from data.iterators import UnusedPointsIterator, ListIterator, PolicyIterator
from data.collectors import DataCollector


# In[8]:


import unittest
import sys
import copy
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
import numpy as np
import warnings


# In[9]:


class TestReplayBuffer(unittest.TestCase):
    
    def test_conflicting_buffer_fill(self):
        bf = BufferFiller()
        rb = bf.fill(size=100)

        val_rb = bf.fill_with_unvisited_states(size=50,visited_buffer=rb)

        rts = set(rb.get_zipped_list())

        vts = set(val_rb.get_zipped_list())

        tst_rb = bf.fill_with_unvisited_states(size=50,visited_buffer=rb+val_rb)

        tst_rb.get_zipped_list()
        tts = set(tst_rb.get_zipped_list())
    
        self.assertTrue(rts.isdisjoint(vts))
        self.assertTrue(rts.isdisjoint(tts))
        self.assertTrue(vts.isdisjoint(tts))

    def test_fill_buffer_with_rollouts(self):
        bf = BufferFiller()
        size=100
        rb = bf.fill(size)
        self.assertEqual(len(rb), size)

    def test_fill_with_list(self):
        bf = BufferFiller()

        rb = bf.fill(size=100)

        rb_list = rb.get_zipped_list()

        rb_copy = bf.fill_with_list(rb_list,size=-1)

        rb_copy_list = rb_copy.get_zipped_list()


        rbs = set(list(rb_list))

        rbcs = set(list(rb_copy_list))
        self.assertEqual(rbs, rbcs)


# In[10]:


class TestIterators(unittest.TestCase):

    def test_unused_points_iterator(self):
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
        self.assertTrue(used_set.isdisjoint(set(unused_set)))
        self.assertEqual(len(used_set.union(unused_set)),(env.grid_size - 2)**2 * num_dirs * num_actions)


    def test_policy_iterator(self):

        pi = PolicyIterator()

        # test continuing where you left off
        last_step = 0
        for i,g in enumerate(pi):
            last_step = pi.env.step_count

            if i == 5:
                break

        for i,g in enumerate(pi):
            if i == 0:
                self.assertEqual(pi.env.step_count, last_step + 1)

        # test full reset
        pi.reset()
        for i,g in enumerate(pi):
            if i == 0:
                self.assertEqual(pi.env.step_count, 1)

    def test_list_iterator(self):
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
        self.assertEqual(i, len(list_of_points) - 1)
        self.assertEqual(test_list, list_of_points)




# In[11]:


class TestCollectors(unittest.TestCase):
    def test_collect_specific_datapoint(self):
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
            self.assertEqual(inp_tup, out_tup)

    def test_collect_data_point_per_policy(self):
        dc = DataCollector()
        for i in range(100):
            trans = dc.collect_data_point_per_policy()
            self.assertEqual(len(trans),11)


# In[12]:


if __name__ == "__main__":
    
    if "ipykernel_launcher" in sys.argv[0]:
        tmp_argv = copy.deepcopy(sys.argv)
        sys.argv = [""]
        try:

            unittest.main()
        except SystemExit:
            pass


        sys.argv = tmp_argv
    else:
        unittest.main()

