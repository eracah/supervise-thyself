
# coding: utf-8

# In[1]:


import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
import torch

from torch import nn

import torch.functional as F

from torch.optim import Adam, RMSprop
import argparse
import sys
import copy
from copy import deepcopy
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

import numpy as np
import time
import json
from pathlib import Path
from functools import partial
from replay_buffer import ReplayMemory, fill_replay_buffer
from base_encoder import Encoder, RawPixelsEncoder
from dqn import get_q_loss, QNet, qpolicy
from inverse_model import InverseModel
from utils import mkstr, initialize_weights, write_ims,                write_to_config_file,                collect_one_data_point,                convert_frame, convert_frames,                do_k_episodes, classification_acc, rollout_iterator
from position_predictor import PosPredictor


# In[2]:


def setup_args():
    tmp_argv = copy.deepcopy(sys.argv)
    test_notebook = False
    if "ipykernel_launcher" in sys.argv[0]:
        sys.argv = [""]
        test_notebook= True
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--env_name",type=str, default='MiniGrid-Empty-16x16-v0'),
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--num_episodes",type=int,default=100)
    parser.add_argument("--resize_to",type=int, nargs=2, default=[84, 84])
    parser.add_argument("--epochs",type=int,default=100000)
    parser.add_argument("--width",type=int,default=32)
    #parser.add_argument("--grayscale",action="store_true")
    parser.add_argument("--batch_norm",action="store_true")
    parser.add_argument("--offline",action="store_true")
    parser.add_argument("--buffer_size",type=int,default=10**6)
    parser.add_argument("--init_buffer_size",type=int,default=50000)
    parser.add_argument("--embed_len",type=int,default=32)
    parser.add_argument("--gamma", type=float,default=0.99)
    parser.add_argument("--action_strings",type=str, nargs='+', default=["forward", "left", "right"])
    parser.add_argument("--no_aux",action="store_true")
    parser.add_argument("--C",type=int,default=10000)
    parser.add_argument("--final_exploration_frame",type=int,default=1000000)
    parser.add_argument("--update_frequency",type=int,default=4)
    parser.add_argument("--noop_max",type=int,default=30)
    parser.add_argument("--reward_clip",action="store_true")
    parser.add_argument("--num_eval_eps",type=int,default=25)
    parser.add_argument("--num_val_batches", type=int, default=100)
    args = parser.parse_args()
    args.resize_to = tuple(args.resize_to)

    sys.argv = tmp_argv
    if test_notebook:
        args.init_buffer_size = 100
        #args.online=True
    mstr = partial(mkstr,args=args)
    output_dirname = ("notebook_" if test_notebook else "") + "_".join([mstr("env_name"),
                                                                        mstr("lr"),
                                                                        mstr("width"),
                                                                        mstr("resize_to")
                                                                       ])
    args.output_dirname = output_dirname
    return args

def setup_dirs_logs(args):
    log_dir = './.logs/%s'%args.output_dirname
    writer = SummaryWriter(log_dir=log_dir)
    write_to_config_file(args.__dict__, log_dir)

    return writer

def setup_replay_buffer(init_buffer_size, with_agent_pos=False):
    print("setting up buffer")
    replay_buffer = ReplayMemory(capacity=args.buffer_size,batch_size=args.batch_size,
                                 with_agent_pos=with_agent_pos)
    fill_replay_buffer(buffer=replay_buffer,
                       size=init_buffer_size,
                       rollout_size=256,
                       env = env,
                       resize_to = args.resize_to,
                       policy= lambda x0: np.random.choice(action_space),
                       with_agent_pos=with_agent_pos
                      )
    print("buffer filled!")
    return replay_buffer

def setup_models():
    encoder = Encoder(in_ch=3,
                      im_wh=args.resize_to,
                      h_ch=args.width,
                      embed_len=args.embed_len,
                      batch_norm=args.batch_norm).to(DEVICE)

    inv_model = InverseModel(encoder=encoder,num_actions=num_actions).to(DEVICE)
    raw_pixel_enc = RawPixelsEncoder(in_ch=3,im_wh=args.resize_to)
    return encoder,inv_model, raw_pixel_enc #,  q_net, target_q_net, 
    

def setup_env():
    env = gym.make(args.env_name)
    if "MiniGrid" in args.env_name:
        action_space = range(3)
    else:
        action_space = list(range(env.action_space.n))
    num_actions = len(action_space)
    return env, action_space, num_actions
    


# In[3]:


def ss_train():
    im_losses, im_accs = [], []
    done = False
    state = env.reset()
    while not done:
        
        im_opt.zero_grad()
        policy = lambda x0: np.random.choice(action_space)
        x0,x1,action,reward,done,x0_coord,x1_coord = collect_one_data_point(convert_fxn=convert_fxn,
                                                      env=env,
                                                      policy=policy,
                                                      get_agent_pos=with_agent_pos)


        replay_buffer.push(x0,x1,action,reward,done,x0_coord,x1_coord)
        x0s,x1s,a_s,rs,dones, x0_coords, x1_coords = replay_buffer.sample(args.batch_size)
        a_pred = inv_model(x0s,x1s)
        im_loss = nn.CrossEntropyLoss()(a_pred,a_s)
        im_losses.append(float(im_loss.data))

        acc = classification_acc(a_pred,y_true=a_s)
        im_accs.append(acc)

        im_loss.backward()
        im_opt.step()
    im_loss, im_acc = np.mean(im_losses), np.mean(im_accs)
    writer.add_scalar("train/loss",im_loss,global_step=episode)
    writer.add_scalar("train/acc",im_acc,global_step=episode)
    print("\tIM-Loss: %8.4f \n\tEpisode IM-Acc: %9.3f%%"%(im_loss, im_acc))

  


# In[4]:


def disentang_eval(encoder):    
    x_dim, y_dim = (env.grid_size, env.grid_size)
    pos_pred = PosPredictor((x_dim,y_dim),embed_len=encoder.embed_len).to(DEVICE)
    opt = Adam(lr=0.1,params=pos_pred.parameters())
    #print("beginning eval...")
    x_accs = []
    y_accs = []
    for i in range(args.num_val_batches):
        pos_pred.zero_grad()
        
        
        batch = replay_buffer.sample(args.batch_size)
        x0s,x1s,a_s,rs,dones,x0_c, x1_c = batch


        x_pred,y_pred = pos_pred(encoder(x0s).detach())

        x_true, y_true = x0_c[:,0],x0_c[:,1]

        loss = nn.CrossEntropyLoss()(x_pred,x_true) + nn.CrossEntropyLoss()(y_pred,y_true)
        
        
        #writer.add_scalar("eval/pos_pred_loss",loss,global_step=i)
        x_accs.append(classification_acc(y_logits=x_pred,y_true=x_true))
        y_accs.append(classification_acc(y_logits=y_pred,y_true=y_true))
        
        loss.backward()
        opt.step()
    x_acc, y_acc = np.mean(x_accs), np.mean(y_accs)
    return x_acc,y_acc


# In[ ]:


def disentang_evals(encoder_dict):
    eval_dict_x = {}
    eval_dict_y = {}
    for name,encoder in encoder_dict.items():
        x_acc, y_acc = disentang_eval(encoder)
        eval_dict_x[name] = x_acc
        eval_dict_y[name] = y_acc
        print("\t%s Position Prediction: \n\t\t x-acc: %9.3f%% \n\t\t y-acc: %9.3f%%"%(name, x_acc, y_acc))
    writer.add_scalars("eval/x_pos_inf_acc",eval_dict_x, global_step=episode)
    writer.add_scalars("eval/y_pos_inf_acc",eval_dict_y, global_step=episode)


# In[ ]:


#train
if __name__ == "__main__":
    with_agent_pos = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    args = setup_args()
    writer = setup_dirs_logs(args)
    env, action_space, num_actions = setup_env()
    convert_fxn = partial(convert_frame, resize_to=args.resize_to)
    replay_buffer = setup_replay_buffer(args.init_buffer_size, with_agent_pos=with_agent_pos)
    encoder, inv_model, raw_pixel_enc = setup_models()
    
    
    im_opt = Adam(lr=args.lr, params=inv_model.parameters())
    global_steps = 0
    for episode in range(args.num_episodes):
        print("episode %i"%episode)
        ss_train()
        if episode % 10 == 0:
            disentang_evals({"inv_model":encoder, "raw_pixel_enc":raw_pixel_enc})
        
        global_steps += 1

