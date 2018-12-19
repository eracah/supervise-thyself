from evaluations.inference_models import InferenceEvalModel
import torch
import numpy as np

class ForwardEvalModel(InferenceEvalModel):
    def __init__(self, forward_predictor, num_classes, args):
        super(ForwardEvalModel,self).__init__(forward_predictor, num_classes, args)
        self.forward_predictor = forward_predictor
    
    # only function that changes
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
        
        