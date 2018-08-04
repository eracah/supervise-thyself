
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
from base_encoder import Encoder, RawPixelsEncoder,RandomLinearProjection,RandomWeightCNN
from dqn import get_q_loss, QNet, qpolicy
from inverse_model import InverseModel
from utils import mkstr, initialize_weights, write_ims,                write_to_config_file,                collect_one_data_point,                convert_frame, convert_frames,                do_k_episodes, classification_acc, rollout_iterator
from evaluation import PosPredictor, Decoder, Headi


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
    raw_pixel_enc = RawPixelsEncoder(in_ch=3,im_wh=args.resize_to).to(DEVICE)
    rand_lin_proj = RandomLinearProjection(embed_len=args.embed_len,im_wh=args.resize_to,in_ch=3).to(DEVICE)
    rand_cnn = Encoder(in_ch=3,
                      im_wh=args.resize_to,
                      h_ch=args.width,
                      embed_len=args.embed_len,
                      batch_norm=args.batch_norm).to(DEVICE)

    return encoder,inv_model, raw_pixel_enc, rand_lin_proj, rand_cnn #,  q_net, target_q_net, 
    

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
    return im_acc

  


# In[4]:


def eval_iter(encoder, num_batches, batch_size=None):
    if not batch_size:
        batch_size = args.batch_size
    for i in range(num_batches):
        batch = replay_buffer.sample(batch_size)
        x0s,x1s,a_s,rs,dones,x0_c, x1_c = batch
        f0 = encoder(x0s).detach()
        f1 = encoder(x1s).detach()
        yield x0s,x1s,f0, f1, a_s, x0_c, x1_c
    


# In[5]:


def train_decoder(encoder, encoder_name, num_decoder_batches):
    decoder = Decoder(in_ch=3,
                      im_wh=args.resize_to,
                      h_ch=args.width,
                      embed_len=args.embed_len).to(DEVICE)
    #num_decoder_batches = 10000
    criterion = nn.MSELoss()
    opt = Adam(lr=0.1,params=decoder.parameters())
    for i , (x0s,x1s,f0, f1, a_s, x0_c, x1_c) in enumerate(eval_iter(encoder,num_decoder_batches)):
        decoder.zero_grad()
        x0_g = decoder(f0)
        loss = criterion(x0_g,x0s)
        writer.add_scalar("eval/qual/decoder/%s_reconst_loss"%(encoder_name),
                          loss, 
                          global_step=i)
        
        orig_grid = make_grid(x0s)
        rec_grid = make_grid(x0_g)
        writer.add_image("orig/%s"%(encoder_name),orig_grid,global_step=i)
        writer.add_image("rec/%s"%(encoder_name),rec_grid,global_step=i)
        loss.backward()
        opt.step()

    return decoder
        
    
    


# In[6]:


def traverse_latent_space(decoder,encoder, name):

    num_dims_to_try = args.embed_len

    num_perturbs = 12


    iterator = eval_iter(encoder,1,batch_size=args.embed_len)
    x,_,z, _, _, _, _ = next(iterator)
    max_perturb = 5 * float(torch.std(z))


    pzs = []
    for dim in enumerate(range(args.embed_len)):

        dim =0 
        z = z[dim].expand(num_perturbs,-1)


        p_mat = torch.zeros_like(z)
        p_vec = torch.linspace(-max_perturb/2,max_perturb/2,num_perturbs)


        p_mat[:,dim] = p_vec
        pz = z + p_mat
        pzs.append(pz)
    all_pzs = torch.cat(pzs).to(DEVICE)

    def plot_gen_z(decoder, zs, cols, encoder_name):
        x_fake = decoder(zs)
        x_grid = make_grid(x_fake,cols)#.numpy().transpose(1,2,0)
        writer.add_image("traverse_latent/name",x_grid,global_step=1)


    plot_gen_z(decoder,all_pzs,num_dims_to_try,name)


# In[7]:


def qual_evals(encoder_dict):
    decoder_dict = {}
    for name,encoder in encoder_dict.items():
        #if "raw" not in name:
        if "inv_model" in name:
            decoder = train_decoder(encoder, name, args.decoder_batches)
            decoder_dict[name] = decoder
            traverse_latent_space(decoder,encoder,name)
        else:
            pass
    
            #TODO make generalized decoder that just reshapes flattened pixels


# In[8]:


def interpolate_latent_space():
    z0 = torch.randn(1, Z_DIM, 1, 1).cuda()

    z1 = torch.randn(1, Z_DIM, 1, 1).cuda()

    zs = []
    for alpha in np.linspace(0,1,11):
        z = alpha*z0 + (1-alpha)*z1
        zs.append(z)
    zs = Variable(torch.cat(zs)).cuda()
    plot_gen_z(zs,11)


# In[9]:


def interpolate_image_space():
    x0 = (lsgan_netG(Variable(z0)).cpu().data + 1) /2

    x1 = (lsgan_netG(Variable(z1)).cpu().data + 1)/2

    xs = []
    for alpha in np.linspace(0,1,11):
        x = alpha*x0 + (1-alpha)*x1
        xs.append(x)

    xs = torch.cat(xs)

    x_grid = make_grid(xs,11)

    x_grid = x_grid.numpy().transpose(1,2,0)
    
    plt.clf()
    plt.figure(figsize=[30,30])
    plt.imshow(x_grid)
    plt.axis("off")
    plt.title("Interpolating in Image Space",fontdict={"size":40})
    plt.show()
    


# In[10]:


def quant_eval(encoder):    
    x_dim, y_dim = (env.grid_size, env.grid_size)
    pos_pred = PosPredictor((x_dim,y_dim),embed_len=encoder.embed_len).to(DEVICE)
    opt = Adam(lr=0.1,params=pos_pred.parameters())
    #print("beginning eval...")
    x_accs = []
    y_accs = []
    
    for x0s,x1s, f0, f1, a_s, x0_c, x1_c in eval_iter(encoder,args.num_val_batches):
        pos_pred.zero_grad()
        x_pred,y_pred = pos_pred(f0)
        x_true, y_true = x0_c[:,0],x0_c[:,1]
        loss = nn.CrossEntropyLoss()(x_pred,x_true) + nn.CrossEntropyLoss()(y_pred,y_true)
        x_accs.append(classification_acc(y_logits=x_pred,y_true=x_true))
        y_accs.append(classification_acc(y_logits=y_pred,y_true=y_true))
        
        loss.backward()
        opt.step()
    x_acc, y_acc = np.mean(x_accs), np.mean(y_accs)
    return x_acc,y_acc

def quant_evals(encoder_dict):
    eval_dict_x = {}
    eval_dict_y = {}
    decoder_dict = {}
    for name,encoder in encoder_dict.items():
        x_acc, y_acc = quant_eval(encoder)
        eval_dict_x[name] = x_acc
        eval_dict_y[name] = y_acc
        print("\t%s Position Prediction: \n\t\t x-acc: %9.3f%% \n\t\t y-acc: %9.3f%%"%(name, x_acc, y_acc))
        
    writer.add_scalars("eval/quant/x_pos_inf_acc",eval_dict_x, global_step=episode)
    writer.add_scalars("eval/quant/y_pos_inf_acc",eval_dict_y, global_step=episode)


# In[11]:


#train
if __name__ == "__main__":
    with_agent_pos = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    args = setup_args()
    writer = setup_dirs_logs(args)
    env, action_space, num_actions = setup_env()
    convert_fxn = partial(convert_frame, resize_to=args.resize_to)
    replay_buffer = setup_replay_buffer(args.init_buffer_size, with_agent_pos=with_agent_pos)
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
        if acc > 99:
            break
        quant_evals({"inv_model":encoder})
        global_steps += 1
    quant_evals(enc_dict)
    qual_evals(enc_dict)
        

