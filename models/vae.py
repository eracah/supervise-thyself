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
        layers = self.setup_decoder_layers(encoder)
        self.decoder = nn.Sequential(*layers)
        
        enc_fc = self.enc_dict["fc"]
        self.fc =  nn.Linear(in_features=enc_fc.out_features,
                            out_features=enc_fc.in_features)


    def setup_decoder_layers(self,encoder):
        bias = encoder.bias

        enc_list = [self.enc_dict[key] for key in self.enc_dict.keys() if "encoder." in key ]
        enc_list.reverse()
        layers = []
        for lay in enc_list:
            if "Conv" in str(lay):
                layer = nn.ConvTranspose2d(in_channels=lay.out_channels,
                                           out_channels=lay.in_channels,
                                           kernel_size=lay.kernel_size,
                                           stride=lay.stride,
                                           padding=lay.padding,
                                           bias=bias,
                                           output_padding=1)
            else:
                layer = copy.deepcopy(lay)
            layers.append(layer)
        return layers
        
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
