import torch
from torch import nn
import torch.functional as F
import numpy as np
from evaluations.utils import classification_acc
from evaluations.linear_model import LinearModel


class InferModel(nn.Module):
    """feeds embeddings from encoder into linear model for inference or prediction depending on what the inputs to the linear model are"""
    def __init__(self, encoder, num_classes, args):
        super(InferModel,self).__init__()
        self.encoder = encoder
        #self.model_type = args.model_type # classifier or regressor
        self.label_name = args.label_name #y_coord or x_coord or other state variables
        self.linear_model = LinearModel(output_len=num_classes,
                                 input_len=encoder.embed_len)
     
    def forward(self,trans):
        x = trans.xs[:,0]
        embeddings = self.encoder(x)
        embeddings = embeddings.detach()
        pred = self.linear_model(embeddings)
        return pred
        
    def get_model_inputs(self,trans):
        
        return embeddings,y

        
    def loss_acc(self,trans):
        y = trans.state_param_dict[self.label_name][:,0].long()
        pred = self.forward(trans)
        
        
        loss = nn.CrossEntropyLoss()(pred,y)
        acc = classification_acc(pred,y)
        return loss, acc


    
    
    