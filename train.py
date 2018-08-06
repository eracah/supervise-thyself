
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
from functools import partial
from replay_buffer import setup_replay_buffer
from base_encoder import Encoder, RawPixelsEncoder,RandomLinearProjection,RandomWeightCNN
#from dqn import get_q_loss, QNet, qpolicy
from inverse_model import InverseModel
from utils import setup_env,mkstr,write_to_config_file,collect_one_data_point, convert_frame, classification_acc
from evaluation import PosPredictor, Decoder, HeadingPredictor,traverse_latent_space


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
    parser.add_argument("--batch_norm",action="store_true")
    parser.add_argument("--buffer_size",type=int,default=10**6)
    parser.add_argument("--init_buffer_size",type=int,default=50000)
    parser.add_argument("--eval_trials",type=int,default=5)
    parser.add_argument("--embed_len",type=int,default=32)
    parser.add_argument("--action_strings",type=str, nargs='+', default=["forward", "left", "right"])
    parser.add_argument("--num_val_batches", type=int, default=100)
    parser.add_argument("--decoder_batches", type=int, default=1000)
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
    


    


# In[3]:


def ss_train():
    im_losses, im_accs = [], []
    done = False
    state = env.reset()
    while not done:
        
        im_opt.zero_grad()
        policy = lambda x0: np.random.choice(action_space)
        transition = collect_one_data_point(convert_fxn=convert_fxn,
                                                      env=env,
                                                      policy=policy,
                                                      with_agent_pos=with_agent_pos,
                                                    with_agent_heading=with_agent_heading)


        done = transition.done
        replay_buffer.push(*transition)
        trans = replay_buffer.sample(args.batch_size)
        a_pred = inv_model(trans.x0,trans.x1)
        im_loss = nn.CrossEntropyLoss()(a_pred,trans.a)
        im_losses.append(float(im_loss.data))

        acc = classification_acc(a_pred,y_true=trans.a)
        im_accs.append(acc)

        im_loss.backward()
        im_opt.step()
    im_loss, im_acc = np.mean(im_losses), np.mean(im_accs)
    writer.add_scalar("train/loss",im_loss,global_step=episode)
    writer.add_scalar("train/acc",im_acc,global_step=episode)
    print("\tIM-Loss: %8.4f \n\tEpisode IM-Acc: %9.3f%%"%(im_loss, im_acc))
    return im_acc

  


# In[4]:


def eval_iter(encoder, num_batches, batch_size=None):
    replay_buffer = setup_rb()
    if not batch_size:
        batch_size = args.batch_size
    for i in range(num_batches):
        batch = replay_buffer.sample(batch_size)
        f0 = encoder(batch.x0).detach()
        f1 = encoder(batch.x1).detach()
        yield batch, f0,f1
    


# In[5]:


def quant_eval(encoder):    
    x_dim, y_dim = (env.grid_size, env.grid_size)
    pos_pred = PosPredictor((x_dim,y_dim),embed_len=encoder.embed_len).to(DEVICE)
    head_pred = HeadingPredictor(num_directions=4, embed_len=encoder.embed_len).to(DEVICE)
    head_opt = Adam(lr=0.1,params=head_pred.parameters())
    opt = Adam(lr=0.1,params=pos_pred.parameters())
    #print("beginning eval...")
    x_accs = []
    y_accs = []
    h_accs = []
    
    for batch,f0,f1 in eval_iter(encoder,args.num_val_batches):
        pos_pred.zero_grad()
        heading_guess = head_pred(f0)
        true_heading = batch.x0_heading
        heading_loss = nn.CrossEntropyLoss()(heading_guess, true_heading)
        h_accs.append(classification_acc(y_logits=heading_guess,y_true=true_heading))
        
        
        
        
        
        x_pred,y_pred = pos_pred(f0)
        x_true, y_true = batch.x0_coords[:,0],batch.x0_coords[:,1]
        loss = nn.CrossEntropyLoss()(x_pred,x_true) + nn.CrossEntropyLoss()(y_pred,y_true)
        x_accs.append(classification_acc(y_logits=x_pred,y_true=x_true))
        y_accs.append(classification_acc(y_logits=y_pred,y_true=y_true))
        
        
        heading_loss.backward()
        head_opt.step()
        loss.backward()
        opt.step()
    x_acc, y_acc, h_acc = np.mean(x_accs), np.mean(y_accs), np.mean(h_accs)
    return x_acc,y_acc, h_acc

def quant_evals(encoder_dict):
    eval_dict_x = {}
    eval_dict_y = {}
    eval_dict_h = {}
    decoder_dict = {}
    for name,encoder in encoder_dict.items():
        x_accs,y_accs,h_accs = [], [], []
        for i in range(args.eval_trials):
            x_acc, y_acc,h_acc = quant_eval(encoder)
            x_accs.append(x_acc)
            y_accs.append(y_acc)
            h_accs.append(h_acc)
        
        eval_dict_x[name] = np.mean(x_accs)
        #eval_dict_x[name]["std"] = np.std(x_accs)
        
        eval_dict_y[name] = np.mean(y_accs)
        #eval_dict_y[name]["std"] = np.std(y_accs)
        
        eval_dict_h[name] = np.mean(h_accs)
        #eval_dict_h[name]["std"] = np.std(h_accs)
        print("\t%s\n\t\tPosition Prediction: \n\t\t\t x-acc: %9.3f%% +- %9.3f \n\t\t\t y-acc: %9.3f%% +- %9.3f"%
              (name, eval_dict_x[name], np.std(x_accs) / np.sqrt(args.eval_trials),
               eval_dict_y[name],np.std(y_accs) / np.sqrt(args.eval_trials)))
        print("\t\tHeading Prediction: \n\t\t\t h-acc: %9.3f%% +- %9.3f"%
              (eval_dict_h[name], np.std(h_accs) / np.sqrt(args.eval_trials)))
        
    writer.add_scalars("eval/quant/x_pos_inf_acc",eval_dict_x, global_step=episode)
    writer.add_scalars("eval/quant/y_pos_inf_acc",eval_dict_y, global_step=episode)
    writer.add_scalars("eval/quant/h_pos_inf_acc",eval_dict_h, global_step=episode)


# In[6]:


#train
if __name__ == "__main__":
    with_agent_pos = True
    with_agent_heading = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    args = setup_args()
    writer = setup_dirs_logs(args)
    env, action_space = setup_env(args.env_name)
    convert_fxn = partial(convert_frame, resize_to=args.resize_to)
    setup_rb = partial(setup_replay_buffer,capacity=args.buffer_size, 
                                        batch_size=args.batch_size, 
                                        init_buffer_size=args.init_buffer_size,
                                        env=env,
                                        action_space=action_space,
                                        resize_to=args.resize_to,
                                        with_agent_pos = with_agent_pos,
                                        with_agent_heading = with_agent_heading)
    replay_buffer = setup_rb()
    encoder, inv_model, raw_pixel_enc, rand_lin_proj, rand_cnn = setup_models()
    enc_dict = {"inv_model":encoder, 
                     "raw_pixel_enc":raw_pixel_enc, 
                     "rand_proj": rand_lin_proj, 
                     "rand_cnn":rand_cnn}
    
    im_opt = Adam(lr=args.lr, params=inv_model.parameters())
    global_steps = 0
    for episode in range(args.num_episodes):
        print("episode %i"%episode)
        acc = ss_train()
        if acc ==100:
            break
        quant_evals({"inv_model":encoder})
        global_steps += 1
    quant_evals(enc_dict)
    #qual_evals(enc_dict,args)
        

