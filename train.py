
# coding: utf-8

# In[1]:


import custom_grids
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
from functools import partial
from replay_buffer import create_and_fill_replay_buffer
from base_encoder import Encoder
from baselines import RawPixelsEncoder,RandomLinearProjection,RandomWeightCNN
from inverse_model import InverseModel
from utils import setup_env,mkstr,write_to_config_file,collect_one_data_point, convert_frame, classification_acc
from evaluation import quant_evals


# In[2]:


#env = gym.make('MiniGrid-Empty-32x32-v0')


# In[3]:


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
    parser.add_argument("--batch_norm",action="store_true")
    parser.add_argument("--buffer_size",type=int,default=10**6)
    parser.add_argument("--init_buffer_size",type=int,default=50000)
    parser.add_argument("--eval_init_buffer_size",type=int,default=1000)
    parser.add_argument("--eval_trials",type=int,default=5)
    parser.add_argument("--embed_len",type=int,default=32)
    parser.add_argument("--action_strings",type=str, nargs='+', default=["forward", "left", "right"])
    parser.add_argument("--num_val_batches", type=int, default=100)
    parser.add_argument("--decoder_batches", type=int, default=1000)
    parser.add_argument("--collect_data",action="store_true")
    args = parser.parse_args()
    args.resize_to = tuple(args.resize_to)

    sys.argv = tmp_argv
    if test_notebook:
        args.init_buffer_size = 100
        #args.online=True
    mstr = partial(mkstr,args=args)
    output_dirname = ("notebook_" if test_notebook else "") + "_".join([mstr("env_name"),
                                                                        mstr("resize_to")
                                                                       ])
    args.output_dirname = output_dirname
    return args

def setup_dirs_logs(args):
    log_dir = './.logs/%s'%args.output_dirname
    writer = SummaryWriter(log_dir=log_dir)
    write_to_config_file(args.__dict__, log_dir)

    return writer



def setup_models():
    encoder = Encoder(in_ch=3,
                      im_wh=args.resize_to,
                      h_ch=args.width,
                      embed_len=args.embed_len,
                      batch_norm=args.batch_norm).to(DEVICE)

    inv_model = InverseModel(encoder=encoder,num_actions=len(action_space)).to(DEVICE)
    raw_pixel_enc = RawPixelsEncoder(in_ch=3,im_wh=args.resize_to).to(DEVICE)
    rand_lin_proj = RandomLinearProjection(embed_len=args.embed_len,im_wh=args.resize_to,in_ch=3).to(DEVICE)
    rand_cnn = Encoder(in_ch=3,
                      im_wh=args.resize_to,
                      h_ch=args.width,
                      embed_len=args.embed_len,
                      batch_norm=args.batch_norm).to(DEVICE)

    return encoder,inv_model, raw_pixel_enc, rand_lin_proj, rand_cnn #,  q_net, target_q_net, 
    


    


# In[4]:


def ss_train(env, args, writer, episode):
    im_losses, im_accs = [], []
    done = False
    state = env.reset()
    i = 0
    while not done:
        
        im_opt.zero_grad()
        if args.collect_data:
            policy = lambda x0: np.random.choice(action_space)
            transition = collect_one_data_point(convert_fxn=convert_fxn,
                                                          env=env,
                                                          policy=policy,
                                                          with_agent_pos=with_agent_pos,
                                                        with_agent_direction=with_agent_direction)


            done = transition.done
            replay_buffer.push(*transition)
        else:
            done = True if i >= 256 else False
        trans = replay_buffer.sample(args.batch_size)
        a_pred = inv_model(trans.x0,trans.x1)
        im_loss = nn.CrossEntropyLoss()(a_pred,trans.a)
        im_losses.append(float(im_loss.data))

        acc = classification_acc(a_pred,y_true=trans.a)
        im_accs.append(acc)

        im_loss.backward()
        im_opt.step()
        i += 1
    im_loss, im_acc = np.mean(im_losses), np.mean(im_accs)
    writer.add_scalar("train/loss",im_loss,global_step=episode)
    writer.add_scalar("train/acc",im_acc,global_step=episode)
    print("\tIM-Loss: %8.4f \n\tEpisode IM-Acc: %9.3f%%"%(im_loss, im_acc))
    return im_loss, im_acc

  


# In[5]:


#train
if __name__ == "__main__":
    with_agent_pos = True
    with_agent_direction = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    args = setup_args()
    writer = setup_dirs_logs(args)
    env, action_space = setup_env(args.env_name)
    convert_fxn = partial(convert_frame, resize_to=args.resize_to)
    setup_rb = partial(create_and_fill_replay_buffer,capacity=args.buffer_size, 
                                        batch_size=args.batch_size, 
                                        env=env,
                                        action_space=action_space,
                                        resize_to=args.resize_to,
                                        with_agent_pos = with_agent_pos,
                                        with_agent_direction = with_agent_direction)
    
    setup_eval_rb = partial(setup_rb,init_buffer_size=args.eval_init_buffer_size)
    
    replay_buffer = setup_rb(init_buffer_size=args.init_buffer_size)
    encoder, inv_model, raw_pixel_enc, rand_lin_proj, rand_cnn = setup_models()
    enc_dict = {"inv_model":encoder, 
                     "raw_pixel_enc":raw_pixel_enc, 
                     "rand_proj": rand_lin_proj, 
                     "rand_cnn":rand_cnn}
    
    im_opt = Adam(lr=args.lr, params=inv_model.parameters())
    global_steps = 0
    for episode in range(args.num_episodes):
        print("episode %i"%episode)
        loss, acc = ss_train(env, args, writer, episode)
        if acc > 1:
            break
        quant_evals({"inv_model":encoder},setup_eval_rb, writer, args, episode)
        global_steps += 1
    eval_dict = quant_evals(enc_dict, setup_eval_rb, writer, args, episode)
    #qual_evals(enc_dict,args)
        

