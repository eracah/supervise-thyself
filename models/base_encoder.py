import torch
from torch import nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self,
                 im_wh=(64,64),
                 in_ch=3,
                 embed_len=32,
                 fc_width = 512,
                 base_enc_name="world_models",
                 **kwargs):
        
        super(Encoder,self).__init__()
        self.base_enc_name = base_enc_name
        self.im_wh = im_wh 
        self.in_ch = in_ch
        self.fc_width = fc_width
        self.embed_len = embed_len
        self.vec = None
        self.encoder = get_encoder(self.base_enc_name, in_ch)
       
        self.fcs = nn.Sequential(nn.Linear(in_features=self.enc_out_shape,
                                           out_features=self.fc_width),
                                 nn.Linear(in_features=self.fc_width,
                                           out_features=self.embed_len)
                                )



    @property
    def enc_out_shape(self):
        return int(np.prod(self.last_im_shape))
    
    @property
    def last_im_shape(self):
        inp_shape = (1,self.in_ch,*self.im_wh)
        a = torch.randn(inp_shape)
        return self.encoder(a).size()[1:]
    

    def forward(self,x):
        fmaps = self.encoder(x)
        self.vec = fmaps.view(fmaps.size(0),-1)
        raw_embedding = self.fcs(self.vec)
        # normalize embedding by its L2 norm
        embedding = raw_embedding # / raw_embedding.norm(p=2,dim=1, keepdim=True)
        return embedding

    
def get_encoder(name, in_ch):
    if name == "world_models":
        enc = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=32,kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4, stride=2, padding=1),
            nn.ReLU())
    elif name == "universe":
        h_ch =32
        enc = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=h_ch,
                      kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            
            nn.Conv2d(in_channels=h_ch, out_channels=h_ch,
                      kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=h_ch, out_channels=h_ch,
                      kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=h_ch, out_channels=h_ch,
                      kernel_size=3, stride=2, padding=1),
            nn.ELU())
    else:
        raise NotImplementedError
        
    return enc