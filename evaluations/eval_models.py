import torch
from torch import nn
import torch.functional as F
import numpy as np
from evaluations.utils import classification_acc

class EvalModel(nn.Module):
    def __init__(self, encoder, num_classes,model_type="classifier", label_name="x_coord"):
        super(EvalModel,self).__init__()
        self.encoder = encoder
        self.model_type = model_type
        self.model = LinearModel(num_classes=num_classes,
                                 embed_len=encoder.embed_len,
                                 model_type=model_type)
        self.label_name = label_name
    def forward(self,x):
        pass
        # embeddings = self.encoder(x)
        # logits = self.clsfr(embeddings)
        # return embeddings, logits
    def loss_acc(self,trans):
        x = trans.xs[:,0]
        y = trans.state_param_dict[self.label_name][:,0]
        #print(y)
        embeddings = self.encoder(x)
        embeddings = embeddings.detach()
        loss,acc = self.model.loss_acc(embeddings,y)
        return loss, acc

    

        
class LinearModel(nn.Module):
    def __init__(self, num_classes=4, embed_len=32, model_type="classifier"): #,lasso_coeff=0.):
        super(LinearModel,self).__init__()
        self.model_type = model_type
        if self.model_type == "classifier":
            self.fc = nn.Linear(in_features=embed_len, out_features=num_classes)
            
        elif self.model_type == "regressor":
            self.fc = nn.Linear(in_features=embed_len,out_features=1)
            
        
    
    def forward(self, embeddings):
        #make sure embedding is detached
        if embeddings.requires_grad:
            print("eeek")
            embeddings = embeddings.detach()
        logits = self.fc(embeddings)
        return logits
    
    def get_loss(self, pred, true):
        loss_xent = 0.
        if self.model_type == "classifier":
            loss = nn.CrossEntropyLoss()(pred,true.long())
        else:
            loss = nn.SmoothL1Loss()(pred,true.float()[:,None])

        return loss
    
    def loss_acc(self,x,y):
        #true = trans.state_param_dict[self.label_name][:,0]
        pred = self.forward(x)
        loss = self.get_loss(pred,y)
        if self.model_type == "classifier":
            acc = classification_acc(pred,y.long())
        else:
            acc = nn.L1Loss()(pred,y.float()[:,None])
        return loss,acc
        
    
    @property
    def importance_matrix(self):
        return self.fc.weight.abs().transpose(1,0).data