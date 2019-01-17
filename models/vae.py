import torch
from torch import nn
import copy
import numpy as np
from models.base_encoder import Encoder

class Decoder(nn.Module):
    def __init__(self,encoder):
        super(Decoder,self).__init__()
        self.enc_dict =  dict(encoder.named_modules())
        self.lsh = encoder.last_im_shape
        self.osh = encoder.enc_out_shape
        self.decoder = self.setup_decoder()        
        self.fc =  nn.Linear(in_features=32,
                            out_features=self.osh)


    def setup_decoder(self):
        decoder = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=256,out_channels=128, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(in_channels=128,out_channels=64, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(in_channels=64,out_channels=32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(in_channels=32,out_channels=3, kernel_size=4, stride=2, padding=1),
                )
        return decoder

    def forward(self,h):
        ht = self.fc(h)
        
        h_im = ht.view(-1, *self.lsh)
        
        im = self.decoder(h_im)
        return im       
        
class VAE(nn.Module):
    def __init__(self,embed_len=32, **kwargs):
        super(VAE, self).__init__()
        self.embed_len = embed_len
        self.encoder = Encoder(embed_len=embed_len, **kwargs)
        
        self.logvar_fc = nn.Linear(in_features=self.encoder.enc_out_shape,
                            out_features=self.embed_len)
            
        self.decoder = Decoder(self.encoder)
    
    def reparametrize(self, mu, logvar):
        if self.training:
            eps = torch.randn(*logvar.size()).to(mu.device)
            std = torch.exp(0.5*logvar)
            z = mu + eps*std
        else:
            z = mu
        return z
        
    def forward(self, x):
        mu = self.encoder(x)
        logvar = self.logvar_fc(self.encoder.vec)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
    

    def get_kl_rec(self,trans):
        x = trans.xs[0]
        x_hat,mu,logvar = self.forward(x)
        num_pixels = int(np.prod(x.size()[1:]))
        kldiv = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar),dim=1) / num_pixels
        rec = torch.sum((x_hat - x)**2,dim=(1,2,3)) / num_pixels
        
        return kldiv, rec
    
    def loss_acc(self,trans):
        acc = None # cuz no accuracy
        kldiv, rec = self.get_kl_rec(trans)
        loss = rec + kldiv
        return loss.mean(),acc
