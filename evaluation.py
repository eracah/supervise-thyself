
# coding: utf-8

# In[6]:


import torch
from torch import nn
import torch.functional as F
import numpy as np
from torch.optim import Adam, RMSprop
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
from utils import setup_env,mkstr,write_to_config_file,collect_one_data_point, convert_frame, classification_acc


# In[ ]:


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# In[5]:


def quant_eval(encoder, setup_rb, num_val_batches, grid_size): 
    x_dim, y_dim = (grid_size, grid_size)
    pos_pred = PosPredictor((x_dim,y_dim),embed_len=encoder.embed_len).to(DEVICE)
    head_pred = HeadingPredictor(num_directions=4, embed_len=encoder.embed_len).to(DEVICE)
    head_opt = Adam(lr=0.1,params=head_pred.parameters())
    opt = Adam(lr=0.1,params=pos_pred.parameters())
    #print("beginning eval...")
    x_accs = []
    y_accs = []
    h_accs = []
    
    for batch,f0,f1 in eval_iter(encoder,num_val_batches, setup_replay_buffer_fn= setup_rb):
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

def quant_evals(encoder_dict, setup_rb, writer, args, episode):
    env = gym.make(args.env_name)
    grid_size = env.grid_size
    strs = ["x","y","h"]
    eval_dict = {k:{"avg_acc":{}, "std":{}, "std_err":{}} for k in strs}
    for name,encoder in encoder_dict.items():
        x_accs,y_accs,h_accs = [], [], []
        for i in range(args.eval_trials):
            x_acc, y_acc,h_acc = quant_eval(encoder,setup_rb,args.num_val_batches, grid_size)
            x_accs.append(x_acc)
            y_accs.append(y_acc)
            h_accs.append(h_acc)
        
        eval_dict["x"]["avg_acc"][name] = np.mean(x_accs)
        eval_dict["y"]["avg_acc"][name] = np.mean(y_accs)
        eval_dict["h"]["avg_acc"][name] = np.mean(h_accs)
        eval_dict["x"]["std"][name] = np.std(x_accs)
        eval_dict["y"]["std"][name] = np.std(y_accs)
        eval_dict["h"]["std"][name] = np.std(h_accs)
        for s in strs:
            eval_dict[s]["std_err"][name] = eval_dict[s]["std"][name] / np.sqrt(args.eval_trials)

        
        print("\t%s\n\t\tPosition Prediction: \n\t\t\t x-acc: %9.3f%% +- %9.3f \n\t\t\t y-acc: %9.3f%% +- %9.3f"%
              (name, eval_dict["x"]["avg_acc"][name], eval_dict["x"]["std_err"][name],
               eval_dict["y"]["avg_acc"][name],eval_dict["y"]["std_err"][name]))
        print("\t\tHeading Prediction: \n\t\t\t h-acc: %9.3f%% +- %9.3f"%
            (eval_dict["h"]["avg_acc"][name], eval_dict["h"]["std_err"][name]))
        
    writer.add_scalars("eval/quant/x_pos_inf_acc",eval_dict["x"]["avg_acc"], global_step=episode)
    writer.add_scalars("eval/quant/y_pos_inf_acc",eval_dict["y"]["avg_acc"], global_step=episode)
    writer.add_scalars("eval/quant/h_pos_inf_acc",eval_dict["h"]["avg_acc"], global_step=episode)
    writer.add_scalars("eval/quant/x_pos_inf_std_err",eval_dict["x"]["std_err"], global_step=episode)
    writer.add_scalars("eval/quant/y_pos_inf_std_err",eval_dict["y"]["std_err"], global_step=episode)
    writer.add_scalars("eval/quant/h_pos_inf_std_err",eval_dict["h"]["std_err"], global_step=episode)
    return eval_dict
    


# In[3]:


def eval_iter(encoder, num_batches,setup_replay_buffer_fn, batch_size=64):
    replay_buffer = setup_replay_buffer_fn()
    for i in range(num_batches):
        batch = replay_buffer.sample(batch_size)
        f0 = encoder(batch.x0).detach()
        f1 = encoder(batch.x1).detach()
        yield batch, f0,f1


# In[3]:


class PosPredictor(nn.Module):
    """Predict the x and y position of the agent given an embedding"""
    def __init__(self,grid_size, embed_len):
        super(PosPredictor,self).__init__()
        x_dim,y_dim = grid_size
        self.fcx = nn.Linear(in_features=embed_len, out_features=x_dim)
        self.fcy = nn.Linear(in_features=embed_len, out_features=y_dim)
    def forward(self, embeddings):
        #make sure embedding is detached
#         if embeddings.requires_grad:
#             embeddings = embeddings.detach()
        x_logits = self.fcx(embeddings)
        y_logits = self.fcy(embeddings)
        return x_logits, y_logits


# In[4]:


class HeadingPredictor(nn.Module):
    """Predict the heading angle of the agent given an embedding"""
    def __init__(self,num_directions, embed_len):
        super(HeadingPredictor,self).__init__()
        self.fc = nn.Linear(in_features=embed_len, out_features=num_directions)
    def forward(self, embeddings):
        #make sure embedding is detached
#         if embeddings.requires_grad:
#             embeddings = embeddings.detach()
        logits = self.fc(embeddings)
        return logits


# In[56]:


class Decoder(nn.Module):
    def __init__(self,im_wh=(84,84),in_ch=3, embed_len=32, h_ch=32):
        super(Decoder, self).__init__()
        #self.fc = nn.Linear(in_features=embed_len, 
         #                     out_features= np.prod(im_wh / 2**num_layers))
        
        
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(in_channels=embed_len,out_channels=h_ch,kernel_size=7,stride=1),
            # nn.BatchNorm2d(h_ch*8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=embed_len,out_channels=h_ch,kernel_size=5,stride=3, padding=1),
            # nn.BatchNorm2d(h_ch*8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=h_ch,out_channels=h_ch,kernel_size=4,stride=2,padding=1),
            # nn.BatchNorm2d(h_ch*4),
            nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=h_ch,out_channels=h_ch,kernel_size=4,stride=2,padding=1),
#             # nn.BatchNorm2d(h_ch*2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=h_ch,out_channels=h_ch,kernel_size=4,stride=2,padding=1),
#             # nn.BatchNorm2d(h_ch),
#             nn.ReLU(),
            nn.ConvTranspose2d(in_channels=h_ch,out_channels=in_ch,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )
        
        
    def forward(self,x):
        #print(self)
        x = x[:,:,None,None]
        return self.upsampling(x)
        
        
        


# In[1]:


def interpolate_latent_space():
    assert False
    z0 = torch.randn(1, Z_DIM, 1, 1).cuda()

    z1 = torch.randn(1, Z_DIM, 1, 1).cuda()

    zs = []
    for alpha in np.linspace(0,1,11):
        z = alpha*z0 + (1-alpha)*z1
        zs.append(z)
    zs = Variable(torch.cat(zs)).cuda()
    plot_gen_z(zs,11)


def interpolate_image_space():
    assert False
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


# In[3]:


def traverse_latent_space(decoder,encoder, name,args):

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


# In[ ]:


def train_decoder(encoder, encoder_name, num_decoder_batches):
    decoder = Decoder(in_ch=3,
                      im_wh=args.resize_to,
                      h_ch=args.width,
                      embed_len=args.embed_len).to(DEVICE)
    #num_decoder_batches = 10000
    criterion = nn.MSELoss()
    opt = Adam(lr=0.1,params=decoder.parameters())
    for i , (batch,f0,f1) in enumerate(eval_iter(encoder,num_decoder_batches)):
        decoder.zero_grad()
        x0_g = decoder(f0)
        loss = criterion(x0_g,batch.x0)
        writer.add_scalar("eval/qual/decoder/%s_reconst_loss"%(encoder_name),
                          loss, 
                          global_step=i)
        
        orig_grid = make_grid(batch.x0)
        rec_grid = make_grid(x0_g)
        writer.add_image("orig/%s"%(encoder_name),orig_grid,global_step=i)
        writer.add_image("rec/%s"%(encoder_name),rec_grid,global_step=i)
        loss.backward()
        opt.step()

    return decoder


# In[5]:


def qual_evals(encoder_dict,args):
    decoder_dict = {}
    for name,encoder in encoder_dict.items():
        #if "raw" not in name:
        if "inv_model" in name:
            decoder = train_decoder(encoder, name, args.decoder_batches)
            decoder_dict[name] = decoder
            traverse_latent_space(decoder,encoder,name,args)
        else:
            pass
    
            #TODO make generalized decoder that just reshapes flattened pixel


# In[ ]:


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

