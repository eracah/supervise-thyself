
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
from utils import convert_frame
import numpy as np
import time
import json
from pathlib import Path
from functools import partial
from replay_buffer import ReplayMemory, fill_replay_buffer
from models import InverseModel, QNet, Encoder
from utils import mkstr, initialize_weights, write_ims, write_to_config_file,collect_one_data_point


# In[2]:


def setup_args():
    tmp_argv = copy.deepcopy(sys.argv)
    test_notebook = False
    if "ipykernel_launcher" in sys.argv[0]:
        sys.argv = [""]
        test_notebook= True
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--env_name",type=str, default='MiniGrid-Empty-6x6-v0'),
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
    parser.add_argument("--with_aux",action="store_true")
    parser.add_argument("--C",type=int,default=10000)
    parser.add_argument("--final_exploration_frame",type=int,default=1000000)
    parser.add_argument("--update_frequency",type=int,default=4)
    parser.add_argument("--noop_max",type=int,default=30)
    parser.add_argument("--reward_clip",action="store_true")
    parser.add_argument("--num_eval_eps",type=int,default=25)
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


# In[3]:


def do_one_update(x0,x1,y, iter_= 0, mode="train"):
    if model.training:
        opt.zero_grad()
    output = model(x0,x1)

    loss = criterion(output,y)
    if model.training:
        loss.backward()
        opt.step()

    action_guess = torch.argmax(output,dim=1)
    acc = (float(torch.sum(torch.eq(y,action_guess)).data) / y.size(0))*100
    #save_incorrect_examples(y,action_guess,x0,x1,iter_)

    return float(loss.data), acc #, inc_actions.data
    
 


# In[4]:


def get_im_loss_acc(x0,x1,a):
    y = a
    a_pred = inv_model(x0,x1)
    im_loss = im_criterion(a_pred,y)
    action_guess = torch.argmax(a_pred,dim=1)
    acc = (float(torch.sum(torch.eq(y,action_guess)).data) / y.size(0))*100
    return im_loss, acc
    


# In[5]:


def get_q_loss(x0,x1,a,r, dones):
    qbootstrap = args.gamma * torch.max(target_q_net(x1).detach(),dim=1)[0]
    # zero out bootstraps for states that are the last state
    qbootsrap = (1-torch.tensor(dones)).cuda().float() * qbootstrap
    y = r + qbootstrap
    #print(dones)
    q_vals = torch.gather(q_net(x0),1,a[:,None])[:,0]
    error = y - q_vals
    error = torch.clamp(error,-1.0,1.0)
    #print(error)
    q_loss = torch.sum(error**2)
    return q_loss


# In[6]:


def e_greedy(q_values,epsilon=0.1):
    r = np.random.uniform(0,1)
    if r < epsilon:
        action = np.random.choice(len(q_values))
    else:
        action = np.argmax(q_values)
    return action


# In[7]:


def setup_models():
    encoder = Encoder(in_ch=3,
                      im_wh=args.resize_to,
                      h_ch=args.width,
                      embed_len=args.embed_len,
                      batch_norm=args.batch_norm).to(DEVICE)
    
    q_net = QNet(encoder=encoder,
                 num_actions=num_actions).to(DEVICE)
    target_q_net = QNet(encoder=encoder,
                 num_actions=num_actions).to(DEVICE)
    target_q_net.load_state_dict(q_net.state_dict())
    

    inv_model = InverseModel(encoder=encoder,num_actions=num_actions).to(DEVICE)
    inv_model.apply(initialize_weights)
    return encoder,q_net, target_q_net, inv_model
    


# In[8]:


def setup_replay_buffer(init_buffer_size):
    print("setting up buffer")
    replay_buffer = ReplayMemory(capacity=args.buffer_size,batch_size=args.batch_size)
    fill_replay_buffer(buffer=replay_buffer,
                       size=init_buffer_size,
                       rollout_size=256,
                       env = env,
                       resize_to = args.resize_to,
                       policy= lambda x0: np.random.choice(action_space),
                      )
    print("buffer filled!")
    return replay_buffer
    


# In[9]:


def setup_env():
    env = gym.make(args.env_name)
    if "MiniGrid" in args.env_name:
        action_space = range(3)
    else:
        action_space = list(range(env.action_space.n))
    num_actions = len(action_space)
    return env, action_space, num_actions
    


# In[10]:


def qpolicy(x0,epsilon=0.1):
    q_values = q_net(x0[None,:])[0].cpu().data.numpy()
    action = e_greedy(q_values,epsilon=epsilon)
    return int(action)
    


# In[11]:


def do_k_episodes(k=1,epsilon=0.1):
    rewards = []
    with torch.no_grad():
        for ep in range(k):
            done = False
            env.reset()
            cum_reward = 0
            while not done:
                _,_,_,reward, done = collect_one_data_point(convert_fxn=convert_fxn,
                                                            env=env,
                                                            policy=partial(qpolicy,epsilon=epsilon))
                cum_reward += float(reward)
            rewards.append(cum_reward)
        return np.mean(rewards), rewards
            
            
    


# In[13]:


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    epsilon = 1
    args = setup_args()
    args.reward_clip = True
    args.with_aux = False
    writer = setup_dirs_logs(args)
    env, action_space, num_actions = setup_env()
    convert_fxn = partial(convert_frame, resize_to=args.resize_to)
    replay_buffer = setup_replay_buffer(args.init_buffer_size)
    
    encoder,    q_net,    target_q_net,    inv_model = setup_models()
    

    im_criterion, q_criterion  = nn.CrossEntropyLoss(), nn.MSELoss()
    
    im_opt = Adam(lr=args.lr, params=inv_model.parameters())
    qopt = Adam(lr=args.lr,params=q_net.parameters())
    #qopt = RMSprop(lr=args.lr,params=q_net.parameters())
    global_steps = 0
    for episode in range(args.num_episodes):
        print("episode %i"%episode)
        done = False
        state = env.reset()
        im_losses = []
        im_accs = []
        qlosses = []
        while not done:
            if global_steps % args.C == 0:
                target_q_net.load_state_dict(q_net.state_dict())
            im_opt.zero_grad()
            qopt.zero_grad()
            
            policy = partial(qpolicy,epsilon=epsilon)
            #uint8 single frame
            x0,x1,a,r, done = collect_one_data_point(convert_fxn=convert_fxn,
                                                          env=env,
                                                          policy=policy)
            if args.reward_clip:
                r = np.clip(r, -1, 1)
            replay_buffer.push(state=x0,next_state=x1,action=a,reward=r,done=done)
            
            #batch of pytorch cuda float tensors 
            x0s,x1s,a_s,rs,dones = replay_buffer.sample(args.batch_size)



            qloss = get_q_loss(x0s,x1s,a_s,rs,dones)
            qloss.backward()
            if args.with_aux:
                im_loss, acc = get_im_loss_acc(x0s,x1s,a_s)
                im_accs.append(acc)
                im_losses.append(float(im_loss.data))
                im_loss.backward()
                im_opt.step()
            if global_steps % args.update_frequency == 0:
                qopt.step()
            qlosses.append(float(qloss.data))
            global_steps += 1
            if global_steps < args.final_exploration_frame:
                epsilon -= (0.9 / args.final_exploration_frame )
            else:
                epsilon = 0.1

        #print(env.step_count)
        #print(epsilon)
        qloss = np.mean(qlosses)
        writer.add_scalar("episode_loss",qloss,global_step=episode)
        #writer.add_scalar("episode_acc",acc,global_step=episode)
        print("\tEpisode QLoss: %8.4f"%(qloss)) 
        if args.with_aux:
            imloss, im_acc = np.mean(im_losses), np.mean(im_accs)
            print("IM-Loss: %8.4f \n\tEpisode IM-Acc: %9.3f%%"%(im_loss, im_acc))
            
        if episode % 5 == 0:
            avg_reward, rewards = do_k_episodes(k=args.num_eval_eps,epsilon=0.0)
            writer.add_scalar("avg_episode_reward",avg_reward,global_step=episode)
            print("\tAverage Episode Reward for Qnet after %i Episodes: %8.4f"%(episode,avg_reward))
        
    
    

