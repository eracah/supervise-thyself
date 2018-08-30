
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
from utils import setup_env, mkstr, write_to_config_file, collect_one_data_point, convert_frame, classification_acc


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

