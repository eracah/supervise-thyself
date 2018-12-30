import torch
import numpy as np
from torch import nn
from evaluations.linear_model import LinearModel

class PredictEvalModel(nn.Module):
    def __init__(self, encoder, args):
        super(PredictEvalModel,self).__init__()
        self.predictor = LinearModel(num_outputs=embed_len,
                                 embed_len=encoder.embed_len)
        self.encoder = encoder
    
  
    def get_model_inputs(self,trans):
        f_preds = self.forward_predictor(trans)
        f_preds = f_preds.detach()
        
        ys = trans.state_param_dict[self.label_name][:,1:]
        return f_preds,ys
    
    def loss_acc(self,trans):
        f_preds,ys = self.get_model_inputs(trans)
        num_steps = ys.shape[-1]
        loss_accs = [self.linear_clsf.loss_acc(f_preds[:,i],ys[:,i]) for i in range(num_steps)]
        losses, accs = zip(*loss_accs)
        torch.stack(losses)
        loss = torch.mean(torch.stack(losses))
        acc = np.mean(accs)
        return loss,acc
        
        