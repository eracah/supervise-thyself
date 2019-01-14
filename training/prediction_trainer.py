from training.inference_trainer import InferenceTrainer
import copy
import torch

class PredictionTrainer(InferenceTrainer):
    def __init__(self, model, args, experiment):
        super(PredictionTrainer, self).__init__(model, args, experiment)
        
    def test(self, test_set):
        self.one_epoch(test_set,mode="test")
        
        self.do_pca_corr(test_set, self.model)
        
    def collect_embeddings_megabatch(self,test_set, encoder):
        fs = []
        ys = []
        for trans in test_set:
            f = encoder(trans)
            frs = f.reshape(f.shape[0]*f.shape[1],f.shape[2])
            fs.append(frs)
            
            # we don't predict the first frame, so don't include the first frame's y info 
            y = trans.state_param_dict[self.label_name][:,1:] 
            yrs = y.reshape(y.shape[0]*y.shape[1])

            ys.append(yrs)

        f = torch.cat(fs)
        y = torch.cat(ys).squeeze()
        return f.detach(), y
        
    