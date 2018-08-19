
# coding: utf-8

# In[2]:


import torch
from torch import nn
import torch.functional as F
import numpy as np
from torch.optim import Adam, RMSprop
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from utils import setup_env, mkstr, write_to_config_file, collect_one_data_point, convert_frame, classification_acc


# In[5]:


def quant_eval(encoder, val_buf, num_val_batches, grid_size): 
    x_dim, y_dim = (grid_size, grid_size)
    pos_pred = PosPredictor((x_dim,y_dim),embed_len=encoder.embed_len).to(DEVICE)
    dir_pred = DirectionPredictor(num_directions=4, embed_len=encoder.embed_len).to(DEVICE)
    dir_opt = Adam(lr=0.1,params=dir_pred.parameters())
    opt = Adam(lr=0.1,params=pos_pred.parameters())
    #print("beginning eval...")
    x_accs = []
    y_accs = []
    h_accs = []
    
    for batch,f0,f1 in eval_iter(encoder,val_buf):
        pos_pred.zero_grad()
        direction_guess = dir_pred(f0)
        true_direction = batch.x0_direction
        direction_loss = nn.CrossEntropyLoss()(direction_guess, true_direction)
        h_accs.append(classification_acc(y_logits=direction_guess,y_true=true_direction))
        
        
        
        
        
        x_pred,y_pred = pos_pred(f0)
        x_true, y_true = batch.x0_coord_x,batch.x0_coord_y
        loss = nn.CrossEntropyLoss()(x_pred,x_true) + nn.CrossEntropyLoss()(y_pred,y_true)
        x_accs.append(classification_acc(y_logits=x_pred,y_true=x_true))
        y_accs.append(classification_acc(y_logits=y_pred,y_true=y_true))
        
        
        direction_loss.backward()
        dir_opt.step()
        loss.backward()
        opt.step()
    x_acc, y_acc, h_acc = np.mean(x_accs), np.mean(y_accs), np.mean(h_accs)
    return x_acc,y_acc, h_acc

def quant_evals(encoder_dict, val_buf, writer, args, episode):
    env = gym.make(args.env_name)
    grid_size = env.grid_size
    strs = ["x","y","h"]
    eval_dict = {k:{"avg_acc":{}, "std":{}, "std_err":{}} for k in strs}
    for name,encoder in encoder_dict.items():
        x_accs,y_accs,h_accs = [], [], []
        for i in range(args.eval_trials):
            x_acc, y_acc,h_acc = quant_eval(encoder,val_buf,args.num_val_batches, grid_size)
            x_accs.append(x_acc)
            y_accs.append(y_acc)
            h_accs.append(h_acc)
        
        eval_dict["x"]["avg_acc"][name] = np.mean(x_accs)
        eval_dict["y"]["avg_acc"][name] = np.mean(y_accs)
        eval_dict["h"]["avg_acc"][name] = np.mean(h_accs)
        eval_dict["x"]["std"][name] = np.std(x_accs)
        eval_dict["y"]["std"][name] = np.std(y_accs)
        eval_dict["h"]["std"][name] = np.std(h_accs)
        for s in strs:
            eval_dict[s]["std_err"][name] = eval_dict[s]["std"][name] / np.sqrt(args.eval_trials)

        
        print("\t%s\n\t\tPosition Prediction: \n\t\t\t x-acc: %9.3f%% +- %9.3f \n\t\t\t y-acc: %9.3f%% +- %9.3f"%
              (name, eval_dict["x"]["avg_acc"][name], eval_dict["x"]["std_err"][name],
               eval_dict["y"]["avg_acc"][name],eval_dict["y"]["std_err"][name]))
        print("\t\tdirection Prediction: \n\t\t\t h-acc: %9.3f%% +- %9.3f"%
            (eval_dict["h"]["avg_acc"][name], eval_dict["h"]["std_err"][name]))
        
    writer.add_scalars("eval/quant/x_pos_inf_acc",eval_dict["x"]["avg_acc"], global_step=episode)
    writer.add_scalars("eval/quant/y_pos_inf_acc",eval_dict["y"]["avg_acc"], global_step=episode)
    writer.add_scalars("eval/quant/h_pos_inf_acc",eval_dict["h"]["avg_acc"], global_step=episode)
    writer.add_scalars("eval/quant/x_pos_inf_std_err",eval_dict["x"]["std_err"], global_step=episode)
    writer.add_scalars("eval/quant/y_pos_inf_std_err",eval_dict["y"]["std_err"], global_step=episode)
    writer.add_scalars("eval/quant/h_pos_inf_std_err",eval_dict["h"]["std_err"], global_step=episode)
    return eval_dict
    


# In[2]:


def eval_iter(encoder,val_buf):
    for batch in val_buf:
        f0 = encoder(batch.x0).detach()
        f1 = encoder(batch.x1).detach()
        yield batch, f0,f1


# In[15]:


class LinearClassifier(nn.Module):
    def __init__(self, num_classes, embed_len,lasso_coeff=0.):
        super(LinearClassifier,self).__init__()
        self.fc = nn.Linear(in_features=embed_len, out_features=num_classes)
        self.lasso_coeff = lasso_coeff
    
    def forward(self, embeddings):
        #make sure embedding is detached
        if embeddings.requires_grad:
            embeddings = embeddings.detach()
        logits = self.fc(embeddings)
        return logits
    
    def get_loss(self,y_pred,y_true):
        loss_xent = nn.CrossEntropyLoss()(y_pred,y_true)
        lasso_term = self.lasso_coeff * self.fc.weight.abs().sum()
        loss = loss_xent + lasso_term
        return loss
    
    @property
    def importance_matrix(self):
        return self.fc.weight.abs()


# In[3]:


class PosPredictor(nn.Module):
    """Predict the x and y position of the agent given an embedding"""
    def __init__(self,grid_size, embed_len,lasso_):
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


class DirectionPredictor(nn.Module):
    """Predict the direction angle of the agent given an embedding"""
    def __init__(self,num_directions, embed_len):
        super(DirectionPredictor,self).__init__()
        self.fc = nn.Linear(in_features=embed_len, out_features=num_directions)
    def forward(self, embeddings):
        #make sure embedding is detached
#         if embeddings.requires_grad:
#             embeddings = embeddings.detach()
        logits = self.fc(embeddings)
        return logits


# In[56]:


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

