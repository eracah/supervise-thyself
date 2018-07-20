
# coding: utf-8

# In[1]:


import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
import torch

from torch import nn

import torch.functional as F

from torch.optim import Adam
import argparse
import sys
import copy
from copy import deepcopy
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from load_data import convert_frame
import numpy as np
import time
import json
from pathlib import Path
from functools import partial
from replay_buffer import ReplayMemory, fill_replay_buffer
from models import InverseModel, QNet, Encoder
from utils import mkstr, initialize_weights, write_ims, write_to_config_file


# In[2]:


def do_one_update(x0,x1,y, iter_= 0, mode="train"):
    x0,x1, y = x0.to(DEVICE), x1.to(DEVICE), y.to(DEVICE)
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
    
 


# In[3]:


def eval_q_loss(x0,a,y):
    q_vals = torch.gather(q_net(x0),1,a[:,None])[:,0]
    q_loss = q_criterion(q_vals,y)
    return q_loss


# In[4]:


def e_greedy(q_values,e=0.1):
    r = np.random.uniform(0,1)
    if r < e:
        action = np.random.choice(len(q_values))
    else:
        action = np.argmax(q_values)
    return action


# In[5]:


# if __name__ == "__main__":
#     q_values = np.random.randn(3)

#     e_greedy(q_values)

#     q_values = torch.randn((3,))

#     e_greedy(q_values)


# In[6]:


def setup_args():
    tmp_argv = copy.deepcopy(sys.argv)
    test_notebook = False
    if "ipykernel_launcher" in sys.argv[0]:
        sys.argv = [""]
        test_notebook= True
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--env_name",type=str, default='MiniGrid-Empty-8x8-v0'),
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--resize_to",type=int, nargs=2, default=[64, 64])
    parser.add_argument("--epochs",type=int,default=100000)
    parser.add_argument("--dataset_size",type=int,default=60000)
    parser.add_argument("--width",type=int,default=32)
    #parser.add_argument("--grayscale",action="store_true")
    parser.add_argument("--batch_norm",action="store_true")
    parser.add_argument("--offline",action="store_true")
    parser.add_argument("--buffer_size",type=int,default=10**6)
    parser.add_argument("--init_buffer_size",type=int,default=1000)
    parser.add_argument("--embed_len",type=int,default=32)
    parser.add_argument("--action_strings",type=str, nargs='+', default=["move_up", "move_down", "move_right", "move_left"])
    args = parser.parse_args()
    args.resize_to = tuple(args.resize_to)
    if args.batch_size > args.dataset_size:
        args.batch_size = args.dataset_size
    sys.argv = tmp_argv
    if test_notebook:
        args.dataset_size = 1000
        #args.online=True
    mstr = partial(mkstr,args=args)
    output_dirname = ("notebook_" if test_notebook else "") + "_".join([mstr("env_name"),
                                                                        mstr("lr"),
                                                                        mstr("width"),
                                                                        mstr("dataset_size"),
                                                                        mstr("resize_to")
                                                                       ])
    args.output_dirname = output_dirname
    return args

def setup_dirs_logs(args):
    log_dir = './.logs/%s'%args.output_dirname
    writer = SummaryWriter(log_dir=log_dir)
    write_to_config_file(args.__dict__, log_dir)

    return writer


# In[7]:


if __name__ == "__main__":
    gamma = 0.9
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    args = setup_args()
    args.action_strings = ["forward", "left", "right"]
    writer = setup_dirs_logs(args)
    env = gym.make(args.env_name)
    if "MiniGrid" in args.env_name:
        action_space = range(3)
#         action_space = create_action_space_minigrid(env=env,
#                                                     list_of_action_strings=args.action_strings)

# #         corner_actions = ["left_or_up", "right_or_up", "left_or_down", "right_or_down"]
# #         label_list = deepcopy(args.action_strings) + corner_actions
# #         num_labels = len(label_list)
    else:
        action_space = list(range(env.action_space.n))
    num_actions = len(action_space)

    replay_buffer = ReplayMemory(capacity=args.buffer_size,batch_size=args.batch_size)
    fill_replay_buffer(replay_buffer,
                       size=args.init_buffer_size, 
                       rollout_size=128,
                       env_name=args.env_name,
                       resize_to = args.resize_to,
                       action_space =action_space)
    
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
    

    model = InverseModel(encoder=encoder,num_actions=num_actions).to(DEVICE)
    model.apply(initialize_weights)
    opt = Adam(lr=args.lr, params=model.parameters())
    criterion = nn.CrossEntropyLoss()
    q_criterion = nn.MSELoss()
    qopt = Adam(lr=args.lr,params=q_net.parameters())
    

    
    num_episodes = 100
    for episode in range(num_episodes):
        print("episode %i"%episode)
        done = False
        state = env.reset()
        obs = convert_frame(env.render("rgb_array"),
                        resize_to=args.resize_to,
                        to_tensor=False)
        losses = []
        accs = []
        while not done:
            q_net.zero_grad()
            x0 = deepcopy(obs)
            x0_tensor = convert_frame(env.render("rgb_array"),
                        resize_to=args.resize_to,
                        to_tensor=True).to(DEVICE)
            q_values = q_net(x0_tensor[None,:])[0].cpu().data.numpy()
            action = e_greedy(q_values)
            obs, reward, done, info = env.step(action)
            if reward > 0:
                print(reward,done)

            obs = convert_frame(env.render("rgb_array"),
                            resize_to=args.resize_to,
                            to_tensor=False)
            a = torch.tensor([action_space.index(action)])
            reward = torch.tensor([reward])
            x1 = deepcopy(obs)
            replay_buffer.push(state=x0,next_state=x1,action=a,reward=reward)
            x0,x1,a,r = replay_buffer.sample(args.batch_size)

            y = r + gamma * torch.max(target_q_net(x1).detach(),dim=1)[0]
            qloss= eval_q_loss(a=a,x0=x0,y=y)
            qloss.backward()
            qopt.step()

            loss, acc = do_one_update(x0,x1,a)
            losses.append(loss)
            accs.append(acc)
        print(env.step_count)
        loss, acc = np.mean(losses), np.mean(accs)
        writer.add_scalar("episode_loss",loss,global_step=episode)
        writer.add_scalar("episode_acc",acc,global_step=episode)
        print("\tEpisode Loss: %8.4f \n\tEpisode Acc: %9.3f%%"%(loss,acc))
        
    
    


# In[ ]:




# if __name__ == "__main__":
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     args = setup_args()
#     writer = setup_dirs_logs(args)

#     if args.offline:
#         trl, vall, tel, label_list = get_tensor_data_loaders(rollout_size=128,action_strings=args.action_strings,
#                                                          env_name=args.env_name, resize_to = args.resize_to,
#                                                         batch_size = args.batch_size, total_examples=args.dataset_size)
#     else:
#         trl = ReplayMemory(capacity=args.buffer_size,batch_size=args.batch_size)
#         trl, label_list = fill_replay_buffer(trl,args.init_buffer_size,rollout_size=128,action_strings=args.action_strings,
#                                                          env_name=args.env_name, resize_to = args.resize_to)
#         vall = ReplayMemory(capacity=args.buffer_size,batch_size=args.batch_size)
#         vall, _ = fill_replay_buffer(vall,args.init_buffer_size,rollout_size=128,action_strings=args.action_strings,
#                                                          env_name=args.env_name, resize_to = args.resize_to)


#     num_actions = len(label_list)
#     in_ch = trl.dataset.tensors[0].size()[1] if args.offline else trl.memory[0].next_state.shape[2]
#     model = InverseModel(in_ch=in_ch,im_wh=args.resize_to,h_ch=args.width,
#                          num_actions=num_actions,batch_norm=args.batch_norm).to(DEVICE)
#     _ = model.apply(initialize_weights)
#     opt = Adam(lr=args.lr, params=model.parameters())
#     criterion = nn.CrossEntropyLoss()


#     if not args.offline:
#         numiter = int(args.dataset_size / args.batch_size)
#     else:
#         numiter = -1
#     for epoch in range(args.epochs):
#         model.train()
#         print("Beginning epoch %i"%(epoch))
#         loss,acc,t = do_epoch(trl,epoch,mode="train",numiter=numiter)
#         print("\tTr Time: %8.4f seconds"% (t))
#         print("\tTr Loss: %8.4f \n\tTr Acc: %9.3f%%"%(loss,acc))
    
#         model.eval()
#         vloss,vacc,t = do_epoch(vall,epoch,mode="val",numiter=numiter)
#         print("\n\tVal Time: %8.4f seconds"% (t))
#         print("\tVal Loss: %8.4f \n\tVal Acc: %9.3f %%"%(vloss,vacc))
    


