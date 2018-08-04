
# coding: utf-8

# In[2]:


import torch
from torch import nn
import torch.functional as F
import numpy as np


# In[3]:


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


# In[4]:


class HeadingPredictor(nn.Module):
    """Predict the heading angle of the agent given an embedding"""
    def __init__(self,num_directions, embed_len):
        super(HeadingPredictor,self).__init__()
        self.fc = nn.Linear(in_features=embed_len, out_features=num_directions)
    def forward(self, embeddings):
        #make sure embedding is detached
#         if embeddings.requires_grad:
#             embeddings = embeddings.detach()
        logits = self.fc(embeddings)
        return logits


# In[56]:


class Decoder(nn.Module):
    def __init__(self,im_wh=(84,84),in_ch=3, embed_len=32, h_ch=32):
        super(Decoder, self).__init__()
        #self.fc = nn.Linear(in_features=embed_len, 
         #                     out_features= np.prod(im_wh / 2**num_layers))
        
        
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(in_channels=embed_len,out_channels=h_ch,kernel_size=7,stride=1),
            # nn.BatchNorm2d(h_ch*8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=embed_len,out_channels=h_ch,kernel_size=5,stride=3, padding=1),
            # nn.BatchNorm2d(h_ch*8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=h_ch,out_channels=h_ch,kernel_size=4,stride=2,padding=1),
            # nn.BatchNorm2d(h_ch*4),
            nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=h_ch,out_channels=h_ch,kernel_size=4,stride=2,padding=1),
#             # nn.BatchNorm2d(h_ch*2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=h_ch,out_channels=h_ch,kernel_size=4,stride=2,padding=1),
#             # nn.BatchNorm2d(h_ch),
#             nn.ReLU(),
            nn.ConvTranspose2d(in_channels=h_ch,out_channels=in_ch,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )
        
        
    def forward(self,x):
        #print(self)
        x = x[:,:,None,None]
        return self.upsampling(x)
        
        
        


# In[ ]:


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

