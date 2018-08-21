
# coding: utf-8

# In[10]:


import torch
from torch import nn
import torch.functional as F
import numpy as np
from torch.optim import Adam, RMSprop
import copy
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from utils import setup_env, mkstr, write_to_config_file, collect_one_data_point, convert_frame, classification_acc

from functools import partial


# In[3]:


def eval_iter(encoder,val_buf):
    for batch in val_buf:
        f0 = encoder(batch.x0).detach()
        f1 = encoder(batch.x1).detach()
        yield batch, f0,f1


# In[4]:


class LinearClassifier(nn.Module):
    def __init__(self, num_classes=4, embed_len=32,lasso_coeff=0.):
        super(LinearClassifier,self).__init__()
        self.fc = nn.Linear(in_features=embed_len, out_features=num_classes)
        #register buffer used to keep lasso_coeff and weights and biases on the same device
        #while keeping requires_grad to false
        self.register_buffer('lasso_coeff', torch.tensor(lasso_coeff))
        
    
    def forward(self, embeddings):
        #make sure embedding is detached
        if embeddings.requires_grad:
            print("eeek")
            embeddings = embeddings.detach()
        logits = self.fc(embeddings)
        return logits
    
    def get_loss(self,pred,true):
        loss_xent = nn.CrossEntropyLoss()(pred,true)
        
        lasso_term = self.fc.weight.abs().sum() * self.lasso_coeff
        loss = loss_xent + lasso_term
        return loss
    
    @property
    def importance_matrix(self):
        return self.fc.weight.abs()


# In[13]:


class QuantEval(object):
    def __init__(self, encoder, encoder_name, val1_buf,val2_buf,test_buf, num_classes, predicted_value_name, args):
        self.encoder = encoder
        self.encoder_name = encoder_name
        # train classifier on val1_buf, hyperparameter tune on val2 buf, test on test buf
        self.val1_buf = val1_buf
        self.val2_buf = val2_buf
        self.test_buf = test_buf
        self.num_classes = num_classes
        self.predicted_value_name = predicted_value_name
        self.args = args
    
    
        self.clsf_template = partial(LinearClassifier,
                            num_classes=self.num_classes,
                            embed_len=self.encoder.embed_len)
        
        self.opt_template = partial(Adam)
        
        self.opt = None
        self.clsf = None
        
        self.iter = eval_iter
        
    def one_iter(self, batch, f0, update_weights=True):
        name = self.predicted_value_name
        if update_weights:
            self.clsf.zero_grad()
        pred = self.clsf(f0)
        true = getattr(batch, name)
        loss = self.clsf.get_loss(pred, true)
        acc = classification_acc(logits=pred,true=true)
        if update_weights:
            loss.backward()
            self.opt.step()
        return loss,acc
    
    def one_epoch(self, buffer,mode="train"):
        update_weights = True if mode=="train" else False
        losses, accs = [], []
        for batch, f0, f1 in self.iter(self.encoder,buffer):
            loss,acc = self.one_iter_update(batch, f0, update_weights=update_weights)
            losses.append(loss)
            accs.append(acc)
        return np.mean(losses), np.mean(accs), self.clsf.state_dict()
        
        
    def train(self,lr, lasso_coeff):
        self.clsf = self.clsf_template(lasso_coeff=lasso_coeff).to(self.args.device)
        self.opt = self.opt_template(params = clsf.parameters(),lr=lr)
        prev_state_dict = None
        for epoch in range(num_epochs):
            self.clsf.train()
            tr_loss, tr_acc, state_dict = self.one_epoch(self.val1_buf, mode="train")
            
      
            self.clsf.eval()
            val_loss, val_acc, _ = self.one_epoch(self.val2_buf, mode="val")
 
            if val_loss > tr_loss:
                break
            prev_state_dict = copy.deepcopy(state_dict)
        return tr_loss, tr_acc, val_loss, val_acc, prev_state_dict
        
    def hyperparameter_tune(self, hyperparam_choices_list):
        best_val_loss = np.inf
        best_hyp = None
        best_state_dict = None
        for lr, lasso_coeff in hyperparam_choices_list:
            tr_loss, tr_acc, val_loss, val_acc, state_dict = self.train(lr,lasso_coeff)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_hyp = [lr,lasso_coeff]
                best_state_dict = copy.deepcopy(state_dict)
        return best_state_dict
    
    def test(self,state_dict):
        # lasso coeff doesn't matter for accuracy
        self.clsf = self.clsf_template(lasso_coeff=0.).to(self.args.device)
        _, acc, _ = self.one_epoch(self.test_buf, mode="test")
        return acc
        
 


# In[35]:


def run_quant_evals(encoder_dict, writer, args):
    lrs = np.random.choice([10**i for i in range(-4,0,1)],size=10)
    lasso_coeff = np.random.choice([i / 10. for i in range(1,11,1)],size=10)
    hyperparams = zip(lrs,lasso_coeff)
    predicted_value_names = ["x0_coord_x","x0_coord_y","x0_direction"]
    eval_dict = {k:{} for k in predicted_value_names}
    for encoder_name,encoder in encoder_dict.items():
        for predicted_value_name in  predicted_value_names:
            qev = QuantEval(encoder, encoder_name, val1_buf,val2_buf,
                            test_buf, num_classes,
                            predicted_value_name,
                            args)
            best_state_dict = qev.hyperparameter_tune(hyperparams)
            acc = qev.test(best_state_dict)
            eval_dict[predicted_value_name][encoder_name] = acc
    
    for predicted_value_name in  predicted_value_names:
        writer.add_scalars("quant/eval/%s",eval_dict[predicted_value_name])
    return eval_dict
        
        


# In[45]:


def quant_eval(encoder, val_buf, grid_size,args): 
    # for minigrid env the grid is nxn but the dimension of places that you can actually go is n-2xn-2
    
    dir_clsf = LinearClassifier(num_classes=4,
                            embed_len=encoder.embed_len).to(args.device)
    x_clsf = LinearClassifier(num_classes=grid_size-2,
                            embed_len=encoder.embed_len).to(args.device)
    y_clsf = LinearClassifier(num_classes=grid_size-2,
                            embed_len=encoder.embed_len).to(args.device)
    
    dir_opt, x_opt, y_opt = Adam(lr=0.1,params=dir_clsf.parameters()),                            Adam(lr=0.1,params=x_clsf.parameters()),                            Adam(lr=0.1,params=y_clsf.parameters())

    x_accs = []
    y_accs = []
    d_accs = []
    
    for batch,f0,f1 in eval_iter(encoder,val_buf):


        dir_clsf.zero_grad()
        dir_pred = dir_clsf(f0)
        dir_true = batch.x0_direction

        dir_loss = dir_clsf.get_loss(dir_pred, dir_true)
        d_accs.append(classification_acc(logits=dir_pred,true=dir_true))
        dir_loss.backward()
        dir_opt.step()
                      
                      
        x_clsf.zero_grad()
        y_clsf.zero_grad()
        x_pred,y_pred = x_clsf(f0), y_clsf(f0)
        # to make it go from 0
        x_true, y_true = batch.x0_coord_x - 1,                        batch.x0_coord_y - 1
        
        

        x_loss = x_clsf.get_loss(pred=x_pred,true=x_true)
        y_loss = y_clsf.get_loss(pred=y_pred,true=y_true)
        
        x_accs.append(classification_acc(logits=x_pred,true=x_true))
        y_accs.append(classification_acc(logits=y_pred,true=y_true))
        x_loss.backward()
        y_loss.backward()
        x_opt.step()
        y_opt.step()
                    
    x_acc, y_acc, d_acc = np.mean(x_accs), np.mean(y_accs), np.mean(d_accs)
    return x_acc,y_acc, d_acc


# In[5]:


def quant_evals(encoder_dict, val_buf, writer, args, episode):
    env = gym.make(args.env_name)
    grid_size = env.grid_size
    strs = ["x","y","d"]
    eval_dict = {k:{"avg_acc":{}, "std":{}, "std_err":{}} for k in strs}
    for name,encoder in encoder_dict.items():
        x_accs,y_accs,d_accs = [], [], []
        for i in range(args.eval_trials):
            x_acc, y_acc,d_acc = quant_eval(encoder,val_buf, grid_size, args)
            x_accs.append(x_acc)
            y_accs.append(y_acc)
            d_accs.append(d_acc)
        
        eval_dict["x"]["avg_acc"][name] = np.mean(x_accs)
        eval_dict["y"]["avg_acc"][name] = np.mean(y_accs)
        eval_dict["d"]["avg_acc"][name] = np.mean(d_accs)
        eval_dict["x"]["std"][name] = np.std(x_accs)
        eval_dict["y"]["std"][name] = np.std(y_accs)
        eval_dict["d"]["std"][name] = np.std(d_accs)
        for s in strs:
            eval_dict[s]["std_err"][name] = eval_dict[s]["std"][name] / np.sqrt(args.eval_trials)

        
        print("\t%s\n\t\tPosition Prediction: \n\t\t\t x-acc: %9.3f%% +- %9.3f \n\t\t\t y-acc: %9.3f%% +- %9.3f"%
              (name, eval_dict["x"]["avg_acc"][name], eval_dict["x"]["std_err"][name],
               eval_dict["y"]["avg_acc"][name],eval_dict["y"]["std_err"][name]))
        print("\t\tdirection Prediction: \n\t\t\t d-acc: %9.3f%% +- %9.3f"%
            (eval_dict["d"]["avg_acc"][name], eval_dict["d"]["std_err"][name]))
        
    writer.add_scalars("eval/quant/x_pos_inf_acc",eval_dict["x"]["avg_acc"], global_step=episode)
    writer.add_scalars("eval/quant/y_pos_inf_acc",eval_dict["y"]["avg_acc"], global_step=episode)
    writer.add_scalars("eval/quant/d_pos_inf_acc",eval_dict["d"]["avg_acc"], global_step=episode)
    writer.add_scalars("eval/quant/x_pos_inf_std_err",eval_dict["x"]["std_err"], global_step=episode)
    writer.add_scalars("eval/quant/y_pos_inf_std_err",eval_dict["y"]["std_err"], global_step=episode)
    writer.add_scalars("eval/quant/d_pos_inf_std_err",eval_dict["d"]["std_err"], global_step=episode)
    return eval_dict
    


# In[3]:


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

