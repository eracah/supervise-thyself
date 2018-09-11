
# coding: utf-8

# In[2]:


from gym_minigrid.envs import EmptyEnv


# In[3]:


from gym_minigrid.register import register


# In[4]:


class EmptyEnv32x32(EmptyEnv):
    def __init__(self):
        super().__init__(size=32)

register(
    id='MiniGrid-Empty-32x32-v0',
    entry_point='data.custom_grids:EmptyEnv32x32'
)


# In[5]:


class EmptyEnv64x64(EmptyEnv):
    def __init__(self):
        super().__init__(size=64)

register(
    id='MiniGrid-Empty-64x64-v0',
    entry_point='data.custom_grids:EmptyEnv64x64'
)


# In[6]:


class EmptyEnv100x100(EmptyEnv):
    def __init__(self):
        super().__init__(size=100)

register(
    id='MiniGrid-Empty-100x100-v0',
    entry_point='data.custom_grids:EmptyEnv100x100'
)

