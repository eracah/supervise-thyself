from torch import nn

class LinearModel(nn.Module):
    def __init__(self,in_feat,out_feat):
        super(LinearModel,self).__init__()
        self.fc = nn.Linear(in_features=in_feat,out_features=out_feat)
    def forward(self,x):
        return self.fc(x)