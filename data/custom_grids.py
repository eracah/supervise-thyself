
# coding: utf-8

# In[5]:


from gym_minigrid.envs import EmptyEnv


# In[6]:


from gym_minigrid.register import register


# In[7]:


class EmptyEnv32x32(EmptyEnv):
    def __init__(self):
        super().__init__(size=32)


# In[ ]:


class EmptyEnv64x64(EmptyEnv):
    def __init__(self):
        super().__init__(size=64)


# In[8]:


register(
    id='MiniGrid-Empty-32x32-v0',
    entry_point='data.custom_grids:EmptyEnv32x32'
)


# In[ ]:


register(
    id='MiniGrid-Empty-64x64-v0',
    entry_point='data.custom_grids:EmptyEnv64x64'
)

