from evaluations.inference_models import InferenceEvalModel

class ForwardEvalModel(InferenceEvalModel):
    def __init__(self, forward_predictor, num_classes, args):
        super(ForwardEvalModel,self).__init__(forward_predictor, num_classes, args)
        self.forward_predictor = forward_predictor
    
    # only function that changes
    def get_model_inputs(self,trans):
        f_preds = self.forward_predictor(trans)
        f_preds = f_preds.detach()
        

        # cuz we looking at da future, so the second one
        y = trans.state_param_dict[self.label_name][:,1:]
        return f_preds,ys
    
    #def loss_acc(self,trans):
        
        