
# coding: utf-8

# In[2]:


import torch
from torch import nn
import torch.functional as F
import numpy as np
from torch.optim import Adam, RMSprop
import copy
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from utils import  mkstr, write_to_config_file,convert_frame, classification_acc

from functools import partial


# In[2]:


def eval_iter(encoder,val_buf):
    for batch in val_buf:
        f0 = encoder(batch.x0).detach()
        f1 = encoder(batch.x1).detach() 
        yield batch, f0,f1


# In[3]:


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
        loss_xent = 0.
        #loss_xent = nn.CrossEntropyLoss()(pred,true)
        
 
        lasso_term = self.fc.weight.abs().sum() * self.lasso_coeff
        loss = loss_xent + lasso_term
        return loss
    
    @property
    def importance_matrix(self):
        return self.fc.weight.abs().transpose(1,0).data


# In[4]:


class QuantEval(object): #it's a god class
    def __init__(self, encoder, encoder_name, val1_buf,val2_buf,test_buf, num_classes, predicted_value_name, args, writer):
        self.encoder = encoder
        self.encoder_name = encoder_name
        # train classifier on val1_buf, hyperparameter tune on val2 buf, test on test buf
        self.val1_buf = val1_buf
        self.val2_buf = val2_buf
        self.test_buf = test_buf
        self.num_classes = num_classes
        self.predicted_value_name = predicted_value_name
        self.args = args
        self.alpha = args.gen_loss_alpha
        self.max_epochs = args.max_quant_eval_epochs
        self.writer=writer
    
    
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
            self.opt.zero_grad()
        pred = self.clsf(f0)
        true = getattr(batch, name)
        loss = self.clsf.get_loss(pred, true)
            
            
        acc = classification_acc(logits=pred,true=true)
        if update_weights:
            loss.backward()
            self.opt.step()
        return float(loss.data),acc
    
    def one_epoch(self, buffer,mode="train"):
        update_weights = True if mode=="train" else False
        losses, accs = [], []
        for batch, f0, f1 in self.iter(self.encoder,buffer):
            loss,acc = self.one_iter(batch, f0, update_weights=update_weights)
            losses.append(loss)
            accs.append(acc)
        return np.mean(losses), np.mean(accs), self.clsf.state_dict()
        
        
    def train(self,lr, lasso_coeff):
        self.clsf = self.clsf_template(lasso_coeff=lasso_coeff).to(self.args.device)
        self.opt = self.opt_template(params = self.clsf.parameters(),lr=lr)
        prev_state_dict = None
        state_dict = self.clsf.state_dict()
        val_loss, tr_loss, min_val_loss = np.inf, np.inf, np.inf
        gen_loss = 0.0
        epoch = 0
        
        while gen_loss <= self.alpha and epoch < self.max_epochs: # this is the GL_alpha early stopping criterion from Prechelt, 1997
            #print(epoch, gen_loss)
            prev_state_dict = copy.deepcopy(state_dict)
            self.clsf.train()
            tr_loss, tr_acc, state_dict = self.one_epoch(self.val1_buf, mode="train")
            
      
            self.clsf.eval()
            val_loss, val_acc, _ = self.one_epoch(self.val2_buf, mode="val")
            if val_loss < min_val_loss:
                min_val_loss = copy.deepcopy(val_loss)

            
            val_min_ratio = ((val_loss+ np.finfo(float).eps) /(min_val_loss + np.finfo(float).eps)) #prevent divide by 0 error
            gen_loss = 100. * (val_min_ratio - 1 )
            
            self.write_acc_loss(tr_loss, tr_acc, val_loss, val_acc, lr, lasso_coeff, epoch)
            if gen_loss == 0. and min_val_loss == 0.:
                break
            epoch+=1

        return tr_loss, tr_acc, val_loss, val_acc, prev_state_dict
    
    def write_acc_loss(self,tr_loss, tr_acc, val_loss, val_acc, lr, lasso_coeff, epoch):
        base_string_enc = "%s/%s"%(self.encoder_name,self.predicted_value_name)
        tr_val_base_string = base_string_enc + "/lr=%0.4f,l1=%0.1f"%(lr, lasso_coeff)
        loss_dict = dict(train=tr_loss, val=val_loss)
        acc_dict = dict(train=tr_acc, val=val_acc)
        self.writer.add_scalars(tr_val_base_string + "/loss",loss_dict,epoch)
        self.writer.add_scalars(tr_val_base_string + "/acc",acc_dict,epoch)
        
        base_string_pred_value = "%s"%(self.predicted_value_name)
        self.writer.add_scalars(base_string_pred_value + "/loss/tr",{"%s_lr=%8.4f,l1=%8.4f"%(self.encoder_name,lr, lasso_coeff):tr_loss}, epoch)
        self.writer.add_scalars(base_string_pred_value + "/loss/val",{"%s_lr=%8.4f,l1=%8.4f"%(self.encoder_name,lr, lasso_coeff):val_loss}, epoch)
        self.writer.add_scalars(base_string_pred_value + "/acc/tr",{"%s_lr=%8.4f,l1=%8.4f"%(self.encoder_name,lr, lasso_coeff):tr_acc}, epoch)
        self.writer.add_scalars(base_string_pred_value + "/acc/val",{"%s_lr=%8.4f,l1=%8.4f"%(self.encoder_name,lr, lasso_coeff):val_acc}, epoch)
        
    def search1d(self,name,hyp_to_vary, train_fxn):
        best_val_loss = np.inf
        best_hyp = None
        best_state_dict = None
        for hyp in hyp_to_vary:
            #print(name, " = ", hyp_to_vary)
            tr_loss, tr_acc, val_loss, val_acc, state_dict = train_fxn(hyp_to_vary)
            #print(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = copy.deepcopy(val_loss)
                best_lr = copy.deepcopy(lr)
                best_state_dict = copy.deepcopy(state_dict)
        
    
    def hyperparameter_tune(self, lrs,lasso_coeffs):
        
        best_val_loss = np.inf
        best_hyp = None
        best_state_dict = None
        for lr in lrs:
            tr_loss, tr_acc, val_loss, val_acc, state_dict = self.train(lr,lasso_coeff=0.0)
            if val_loss < best_val_loss:
                best_val_loss = copy.deepcopy(val_loss)
                best_lr = copy.deepcopy(lr)
                #best_state_dict = copy.deepcopy(state_dict)
        

        
        best_val_loss = np.inf
        best_hyp = None
        best_state_dict = None
        for lasso_coeff in lasso_coeffs:
            tr_loss, tr_acc, val_loss, val_acc, state_dict = self.train(lr=best_lr,lasso_coeff=lasso_coeff)
            #print(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = copy.deepcopy(val_loss)
                best_l1_coeff = copy.deepcopy(lasso_coeff)
                best_state_dict = copy.deepcopy(state_dict)
                
        #print("best lr = ",best_lr, " best l1_coeff = ", best_l1_coeff)
        
                
        return best_state_dict, best_lr, best_l1_coeff
    
    def test(self,state_dict):
        self.clsf = self.clsf_template(lasso_coeff=0.).to(self.args.device)
        self.clsf.load_state_dict(state_dict)
        inform = self._test_informativeness()
        disent  = self._test_disentanglement()
        compl = self._test_completeness()
        return disent, compl, inform
        
        
    
    def _test_informativeness(self):
        _, acc, _ = self.one_epoch(self.test_buf, mode="test")
        return acc
    
    def _test_disentanglement(self):
        R = self.clsf.importance_matrix
        K = R.size(1)

        P = R / R.sum(dim=1,keepdim=True)

        # change of base Rule
        log_kP = torch.log10(P) / torch.log10(K * torch.ones_like(R))

        H = -(P * log_kP).sum(dim=1)

        D = 1 - H


        ro = R.sum(dim=1) / R.sum()

        disentanglement = (ro * D).sum()
        return float(disentanglement)
    
    
    def _test_completeness(self):
        R = self.clsf.importance_matrix
        D = R.size(0)

        Ps = R / R.sum(dim=0,keepdim=True)

        # change of base Rule
        log_dPs = torch.log10(Ps) / torch.log10(D * torch.ones_like(R))

        Hd = -(Ps * log_dPs).sum(dim=0)

        C = 1 - Hd
        return float(C.mean())

    
    
        
 


# In[5]:


class QuantEvals(object):
    def __init__(self, val1_buf, val2_buf, test_buf, writer, grid_size,num_directions, args):
        self.val1_buf = val1_buf
        self.val2_buf = val2_buf
        self.test_buf = test_buf
        self.args = args
        self.writer = writer
        self.predicted_value_names = ["x0_coord_x","x0_coord_y","x0_direction"]
        self.class_dict = dict(zip(self.predicted_value_names, [grid_size,grid_size,num_directions]))
    
    def get_hyperparam_settings(self):
        lrs = [10**i for i in range(-5,0,1)]
        lasso_coeffs = np.linspace(0,1,4)
        return lrs, lasso_coeffs
    
    def run_evals(self, encoder_dict):
        lrs, l1_coeffs = self.get_hyperparam_settings()
        eval_dict = {k:{} for k in self.predicted_value_names}
        for predicted_value_name in  self.predicted_value_names:
            self.print_latex_table_header(predicted_value_name)
            for encoder_name,encoder in encoder_dict.items():                
                qev = QuantEval(encoder, 
                                encoder_name, 
                                self.val1_buf,
                                self.val2_buf,
                                self.test_buf, 
                                num_classes=self.class_dict[predicted_value_name],
                                predicted_value_name=predicted_value_name,
                                args=self.args, writer=self.writer)

                best_state_dict, best_lr, best_l1_coeff = qev.hyperparameter_tune(lrs, l1_coeffs)
                disent, compl, inform = qev.test(best_state_dict)
                eval_dict[qev.predicted_value_name][qev.encoder_name] = inform
                self.print_latex_table_row(qev.encoder_name,best_lr, best_l1_coeff, disent, compl, inform)
            self.print_latex_table_footer()
        for predicted_value_name in self.predicted_value_names:
            self.writer.add_scalars("test/%s"%(predicted_value_name),eval_dict[predicted_value_name])
        return eval_dict
    
    
    
    def print_latex_table_header(self, predicted_value_name ):
        print("\\begin{table}[h]")
        print("\caption{Usefulness results for %s for %s using lasso classifier}"%(predicted_value_name.replace("_","-"), self.args.env_name))
        print("\label{sample-table}")
        print("\\begin{center}")
        print("\\begin{tabular}{llllll}")
        print("\multicolumn{1}{c}{\\bf Code}  &\multicolumn{1}{c}{\\bf Best LR} &\multicolumn{1}{c}{\\bf Best L1} &\multicolumn{1}{c}{\\bf Disent.} &\multicolumn{1}{c}{\\bf Compl.} &\multicolumn{1}{c}{\\bf Inform.}")
        print(" \\\ \hline \\\ ")
        
    def print_latex_table_row(self,encoder_name,best_lr, best_l1_coeff, disent, compl, inform):
        print("%s & %s & %0.2f & %0.2f & %0.2f & %0.2f \\\ "%(encoder_name.replace("_","-"),str(best_lr), best_l1_coeff, disent, compl, inform))
        
    def print_latex_table_footer(self):
        print("\hline")
        print("\end{tabular}")
        print("\end{center}")
        print("\end{table}")

            
        


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

