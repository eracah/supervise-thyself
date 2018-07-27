
# coding: utf-8

# In[22]:


import torch
from torch import nn
import torch.functional as F


# In[138]:


class PosPredictor(nn.Module):
    """Predict the x and y position of the agent given an embedding"""
    def __init__(self,grid_size, embed_len):
        super(PosPredictor,self).__init__()
        x_dim,y_dim = grid_size
        self.fcx = nn.Linear(in_features=embed_len, out_features=x_dim)
        self.fcy = nn.Linear(in_features=embed_len, out_features=y_dim)
    def forward(self, embeddings):
        #make sure embedding is detached
#         if embeddings.requires_grad:
#             embeddings = embeddings.detach()
        x_logits = self.fcx(embeddings)
        y_logits = self.fcy(embeddings)
        return x_logits, y_logits
        


# In[136]:


# if __name__ == "__main__":
#     import gym
#     from gym_minigrid.register import env_list
#     from gym_minigrid.minigrid import Grid
#     from matplotlib import pyplot as plt
#     %matplotlib inline

#     embed_len = 32
#     env_name = "MiniGrid-Empty-6x6-v0"
#     env = gym.make(env_name)
#     env.reset()
#     env.step(2)
#     #print(env.agent_pos)
#     #plt.imshow(env.render("rgb_array"))
#     x_dim, y_dim = env.grid_size, env.grid_size

#     pp = PosPredictor((x_dim, y_dim),embed_len=embed_len)

#     y_truth = torch.randint(0,6,size=(128,)).long()

#     x_truth = torch.randint(0,6,size=(128,)).long()

#     x_g, y_g = pp(embedding)

#     cls_crt = nn.CrossEntropyLoss()

#     from base_encoder import Encoder

#     enc = Encoder()

#     ims = torch.randn((128,3,64,64))

#     embeddings = enc(ims)

#     em = embeddings.detach()

#     x_g, y_g = pp(em)

#     loss = cls_crt(x_g,x_truth) + cls_crt(y_g,y_truth)

#     loss.backward()

