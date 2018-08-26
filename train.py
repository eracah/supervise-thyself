
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
from replay_buffer import BufferFiller
from base_encoder import Encoder
from baselines import RawPixelsEncoder,RandomLinearProjection,RandomWeightCNN
from inverse_model import InverseModel
from utils import mkstr,write_to_config_file,                collect_one_data_point, convert_frame, classification_acc
from quant_evaluation import QuantEvals

#env = gym.make('MiniGrid-Empty-32x32-v0')

def parse_mg(name):
    return name.split("-")[2].split("x")[0]

def setup_args():
    tmp_argv = copy.deepcopy(sys.argv)
    test_notebook = False
    if "ipykernel_launcher" in sys.argv[0]:
        sys.argv = [""]
        test_notebook= True
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lasso_coeff", type=float, default=0.1)
    parser.add_argument("--max_quant_eval_epochs", type=int, default=25)
    parser.add_argument("--gen_loss_alpha", type=float, default=0.4)
    parser.add_argument("--env_name",type=str, default='MiniGrid-Empty-6x6-v0'),
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--val_batch_size",type=int,default=32)
    parser.add_argument("--num_episodes",type=int,default=100)
    parser.add_argument("--resize_to",type=int, nargs=2, default=[84, 84])
    parser.add_argument("--epochs",type=int,default=100000)
    parser.add_argument("--hidden_width",type=int,default=32)
    parser.add_argument("--batch_norm",action="store_true")
    parser.add_argument("--buffer_size",type=int,default=10**6)
    parser.add_argument("--init_buffer_size",type=int,default=50000)
    parser.add_argument("--eval_init_buffer_size",type=int,default=1000)
    parser.add_argument("--eval_trials",type=int,default=5)
    parser.add_argument("--embed_len",type=int,default=32)
    parser.add_argument("--action_strings",type=str, nargs='+', default=["forward", "left", "right"])
    parser.add_argument("--decoder_batches", type=int, default=1000)
    parser.add_argument("--collect_data",action="store_true")
    args = parser.parse_args()
    args.resize_to = tuple(args.resize_to)

    sys.argv = tmp_argv
    if test_notebook:
        args.batch_size = 5
        #args.online=True
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    mstr = partial(mkstr,args=args)
    output_dirname = ("nb_" if test_notebook else "") + "_".join(["e%s"%parse_mg(args.env_name),
                                                                        "r%i"%(args.resize_to[0])
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
                      h_ch=args.hidden_width,
                      embed_len=args.embed_len,
                      batch_norm=args.batch_norm).to(args.device)

    inv_model = InverseModel(encoder=encoder,num_actions=len(action_space)).to(args.device)
    raw_pixel_enc = RawPixelsEncoder(in_ch=3,im_wh=args.resize_to).to(args.device)
    rand_lin_proj = RandomLinearProjection(embed_len=args.embed_len,im_wh=args.resize_to,in_ch=3).to(args.device)
    rand_cnn = Encoder(in_ch=3,
                      im_wh=args.resize_to,
                      h_ch=args.hidden_width,
                      embed_len=args.embed_len,
                      batch_norm=args.batch_norm).to(args.device)

    return encoder,inv_model, raw_pixel_enc, rand_lin_proj, rand_cnn #,  q_net, target_q_net, 
    

def setup_tr_val_val_test(env, policy, convert_fxn, tot_examples):

    
    
    bf = BufferFiller(convert_fxn=convert_fxn, env=env, policy=policy)
    print("creating tr_buf")
    tr_buf = bf.create_and_fill(size=int(0.7*tot_examples),
                                batch_size=args.batch_size)
    print(len(tr_buf))
    print("creating val1_buf")
    val1_buf = bf.create_and_fill(size=int(0.1*tot_examples),
                                batch_size=args.val_batch_size,
                                conflicting_buffer=tr_buf)
    print(len(val1_buf))
    print("creating val2_buf")
    val2_buf = bf.create_and_fill(size=int(0.1*tot_examples),
                                batch_size=args.val_batch_size,
                                conflicting_buffer=tr_buf + val1_buf)
    print(len(val2_buf))
    print("creating test_buf")
    test_buf = bf.create_and_fill(size=int(0.1*tot_examples),
                                batch_size=args.val_batch_size,
                                conflicting_buffer=tr_buf+val1_buf + val2_buf)
    print(len(test_buf))
    return tr_buf, val1_buf, val2_buf, test_buf
    
    

def setup_env(env_name):
    env = gym.make(env_name)
    if "MiniGrid" in env_name:
        action_space = range(3)
        grid_size = env.grid_size - 2
        num_directions = 4
        tot_examples = grid_size**2 * num_directions * len(action_space)
    else:
        action_space = list(range(env.action_space.n))
        grid_size = None
        num_directions = None
        tot_exampls = None
    num_actions = len(action_space)
    return env, action_space, grid_size, num_directions, tot_examples

def ss_train(writer, episode, tr_buf):
    im_losses, im_accs = [], []
    done = False
    state = env.reset()
    for trans in tr_buf:
        a_pred = inv_model(trans.x0,trans.x1)
        im_loss = nn.CrossEntropyLoss()(a_pred,trans.a)
        im_losses.append(float(im_loss.data))

        acc = classification_acc(logits=a_pred,true=trans.a)
        im_accs.append(acc)

        im_loss.backward()
        im_opt.step()
        #i += 1
    im_loss, im_acc = np.mean(im_losses), np.mean(im_accs)
    writer.add_scalar("inv_model/tr_loss",im_loss,global_step=episode)
    writer.add_scalar("inv_model/tr_acc",im_acc,global_step=episode)
    print("\tIM-Loss: %8.4f \n\tEpisode IM-Acc: %9.3f%%"%(im_loss, im_acc))
    return im_loss, im_acc

  

def train_inv_model():
    im_opt = Adam(lr=args.lr, params=inv_model.parameters())

    global_steps = 0
    acc = 0 
    episode = 0
    while acc < 99.:
        print("episode %i"%episode)
        loss, acc = ss_train(writer, episode, tr_buf)
        episode += 1
        break


# In[2]:


#train
if __name__ == "__main__":
    
    args = setup_args()
    writer = setup_dirs_logs(args)
    env, action_space, grid_size, num_directions, tot_examples = setup_env(args.env_name)
    convert_fxn = partial(convert_frame, resize_to=args.resize_to)
    policy=lambda x0: np.random.choice(action_space)
    tr_buf, val1_buf, val2_buf, test_buf = setup_tr_val_val_test(env, policy, convert_fxn, tot_examples)
    
    
    
    encoder, inv_model, raw_pixel_enc, rand_lin_proj, rand_cnn = setup_models()
    enc_dict = {"rand_cnn":rand_cnn, "rand_proj":rand_lin_proj} #"raw_pix":raw_pixel_enc,"inv_model":encoder }
    
    #train_inv_model()
    qevs = QuantEvals(val1_buf, val2_buf, test_buf, writer,
               grid_size,num_directions, args)

    eval_dict = qevs.run_evals(enc_dict)

        

