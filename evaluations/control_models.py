import torch
from torch import nn
import cma
import gym
import numpy as np

class ControlModel(nn.Module):
    def __init__(self,parameters=None,embed_len=32, num_actions=3,  **kwargs):
        super(C,self).__init__()
        self.embed_len = embed_len
        self.num_actions = num_actions
        self.fc = nn.Linear(in_features=embed_len,out_features=num_actions)
        if parameters is not None:
            weight_len = np.prod(self.fc.weight.size())
            weight = parameters[:weight_len]
            weight = np.resize(weight,self.fc.weight.size())
            bias = parameters[weight_len:]
            bias = np.resize(bias,self.fc.bias.size())
            self.fc.weight.requires_grad_(False)
            self.fc.bias.requires_grad_(False)
            self.fc.weight.set_(torch.from_numpy(weight).float())
            self.fc.bias.set_(torch.from_numpy(bias).float())
    
#     def posprocess_output(self,raw_output):
#         raw_steer, raw_gas, raw_brake = raw_output[0],raw_output[1],raw_output[2]

#         steer = F.tanh(raw_steer) # between -1 and 1

#         gas = F.softplus(raw_gas) # between 0 and 1

#         brake = F.softplus(raw_brake) # between 0 and 1
#         action = torch.cat((steer,gas,brake))
#         return action
    
    def postprocess_output(self, raw_output):
        action = torch.argmax(raw_output)
        return action
        
    
    def forward(self,z):
        z = z.squeeze()
        raw_output = self.fc(z)
        action = self.postprocess_output(raw_output)

        return action
            
        

        