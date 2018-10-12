import torch

def classification_acc(logits,true):
    guess = torch.argmax(logits,dim=1)
    acc = (float(torch.sum(torch.eq(true,guess)).data) / true.size(0))
    return acc
