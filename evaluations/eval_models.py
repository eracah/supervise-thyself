import torch
from torch import nn
import torch.functional as F
import numpy as np
from evaluations.utils import classification_acc

class EvalModel(nn.Module):
    def __init__(self, encoder, num_classes, label_name="x_coord"):
        super(EvalModel,self).__init__()
        self.encoder = encoder
        self.clsfr = LinearClassifier(num_classes=num_classes, embed_len=encoder.embed_len)
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
        loss,acc = self.clsfr.loss_acc(embeddings,y)
        return loss, acc
        
        
class LinearClassifier(nn.Module):
    def __init__(self, num_classes=4, embed_len=32): #,lasso_coeff=0.):
        super(LinearClassifier,self).__init__()
        self.fc = nn.Linear(in_features=embed_len, out_features=num_classes)
        # #register buffer used to keep lasso_coeff and weights and biases on the same device
        # #while keeping requires_grad to false
        # self.register_buffer('lasso_coeff', torch.tensor(lasso_coeff))
        
    
    def forward(self, embeddings):
        #make sure embedding is detached
        if embeddings.requires_grad:
            print("eeek")
            embeddings = embeddings.detach()
        logits = self.fc(embeddings)
        return logits
    
    def get_loss(self,pred,true):
        loss_xent = 0.
        loss_xent = nn.CrossEntropyLoss()(pred,true)
        #lasso_term = self.fc.weight.abs().sum() * self.lasso_coeff
        loss = loss_xent #+ lasso_term
        return loss
    
    def loss_acc(self,x,y):
        #true = trans.state_param_dict[self.label_name][:,0]
        pred = self.forward(x)
        loss = self.get_loss(pred,y)
        acc = classification_acc(pred,y)
        return loss,acc
        
    
    @property
    def importance_matrix(self):
        return self.fc.weight.abs().transpose(1,0).data