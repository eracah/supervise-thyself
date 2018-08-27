
# coding: utf-8

# In[8]:


from base_encoder import Encoder
from utils import setup_env
from replay_buffer import BufferFiller
import argparse
from inverse_model import InverseModel
from utils import convert_frame, classification_acc, mkstr, setup_dirs_logs, parse_minigrid_env_name
import argparse
import sys
import copy
import torch
from functools import partial
from torch import nn
from torch.optim import Adam, RMSprop
import numpy as np


# In[9]:


def setup_args():
    tmp_argv = copy.deepcopy(sys.argv)
    test_notebook = False
    if "ipykernel_launcher" in sys.argv[0]:
        sys.argv = [""]
        test_notebook= True
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--lasso_coeff", type=float, default=0.1)
    parser.add_argument("--env_name",type=str, default='MiniGrid-Empty-6x6-v0'),
    parser.add_argument("--resize_to",type=int, nargs=2, default=[84, 84])
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--epochs",type=int,default=100000)
    parser.add_argument("--hidden_width",type=int,default=32)
    parser.add_argument("--embed_len",type=int,default=32)
    args = parser.parse_args()
    args.resize_to = tuple(args.resize_to)

    sys.argv = tmp_argv
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    mstr = partial(mkstr,args=args)
    output_dirname = ("nb_" if test_notebook else "") + "_".join(["inv_model_e%s"%parse_minigrid_env_name(args.env_name),
                                                                        "r%i"%(args.resize_to[0])
                                                                       ])
    args.output_dirname = output_dirname
    return args


# In[10]:


def ss_train(model, opt, writer, episode, tr_buf):
    im_losses, im_accs = [], []
    done = False
    for trans in tr_buf:
        opt.zero_grad()
        a_pred = model(trans.x0,trans.x1)
        im_loss = nn.CrossEntropyLoss()(a_pred,trans.a)
        im_losses.append(float(im_loss.data))

        acc = classification_acc(logits=a_pred,true=trans.a)
        im_accs.append(acc)

        im_loss.backward()
        opt.step()

    im_loss, im_acc = np.mean(im_losses), np.mean(im_accs)
    writer.add_scalar("inv_model/tr_loss",im_loss,global_step=episode)
    writer.add_scalar("inv_model/tr_acc",im_acc,global_step=episode)
    print("\tIM-Loss: %8.4f \n\tEpisode IM-Acc: %9.3f%%"%(im_loss, 100*im_acc))
    return im_loss, im_acc

def get_tr_buf(args, env, action_space, tot_examples):
    convert_fxn = partial(convert_frame, resize_to=args.resize_to)
    policy=lambda x0: np.random.choice(action_space)
    
    bf = BufferFiller(convert_fxn=convert_fxn, env=env, policy=policy)
    tr_buf = bf.create_and_fill(size=int(0.7*tot_examples),
                                batch_size=args.batch_size)
    return tr_buf

        
def setup_model(args, action_space):
    encoder = Encoder(in_ch=3,
                      im_wh=args.resize_to,
                      h_ch=args.hidden_width,
                      embed_len=args.embed_len).to(args.device)

    inv_model = InverseModel(encoder=encoder,num_actions=len(action_space)).to(args.device)
    return inv_model

def train_inv_model(args, writer):
    env, action_space, grid_size, num_directions, tot_examples = setup_env(args.env_name)
    inv_model = setup_model(args, action_space)
    im_opt = Adam(lr=args.lr, params=inv_model.parameters())
    tr_buf = get_tr_buf(args, env, action_space, tot_examples)
    global_steps = 0
    acc = 0 
    episode = 0
    while acc < 0.99:
        print("episode %i"%episode)
        loss, acc = ss_train(inv_model, im_opt, writer, episode, tr_buf)
        episode += 1
    


# In[11]:


if __name__ == "__main__":
    args = setup_args()
    writer = setup_dirs_logs(args)
    train_inv_model(args, writer)

