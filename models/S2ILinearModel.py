# create a linear model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import torchvision.models as models
import sys

__all__ = [
        "S2ILinearModel",
        "S2ILayer2Model",
        "S2ILayer2Model_IAS",
        ]

class S2ILinearModel(nn.Module):
    
    def __init__(self,img_dims,att_dims,bias=True,ClassifierType='Cos'):
        super(S2ILinearModel,self).__init__()
        self.w = nn.Linear(att_dims,img_dims,bias=bias)
        self.ClassifierType=ClassifierType
        self.ReLU = torch.nn.ReLU()
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0).cuda(),requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10).cuda(),requires_grad=True)

    def forward(self, AttM, input, label=None, TrainOrTest='Train'):

        classifier = self.w(AttM)
        classifier = self.ReLU(classifier)
        x = F.normalize(input)
        classifier = F.normalize(classifier)
        out = self.scale_cls*(torch.mm(x,classifier.transpose(1,0))+self.bias)

        return out
        

class S2ILayer2Model(nn.Module):
    
    def __init__(self,img_dims,att_dims,bias=True,ClassifierType='Cos'):
        super(S2ILayer2Model,self).__init__()
        self.L1 = nn.Linear(att_dims,1600,bias=bias)
        self.L2 = nn.Linear(1600,img_dims,bias=bias)
        self.ClassifierType=ClassifierType
        self.ReLU1 = torch.nn.ReLU()
        self.ReLU2 = torch.nn.ReLU()

        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0).cuda(),requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10).cuda(),requires_grad=True)

    def forward(self, AttM, input, label=None, TrainOrTest='Train',clsGroup=None):
        
        W1 = self.L1(AttM)
        W1 = self.ReLU1(W1)
        W2 = self.L2(W1)
        classifier = self.ReLU2(W2)
        if self.ClassifierType == 'Cos':
            x = F.normalize(input)
            classifier = F.normalize(classifier)
        out = self.scale_cls*(torch.mm(x,classifier.transpose(1,0))+self.bias)

        if TrainOrTest=='Train':
            return out
        else:
            return out,out/self.scale_cls


class S2ILayer2Model_IAS(nn.Module):
    
    def __init__(self,img_dims,att_dims,bias=True,ClassifierType='Cos',tmp=5):
        super(S2ILayer2Model_IAS,self).__init__()

        self.L1 = nn.Linear(att_dims,1600,bias=bias)
        self.L2 = nn.Linear(1600,img_dims,bias=bias)
        self.ClassifierType=ClassifierType
        self.ReLU1 = torch.nn.ReLU()
        self.ReLU2 = torch.nn.ReLU()
        self.tmp = tmp
        self.ReLU = torch.nn.ReLU()

        self.img_guaid = nn.Linear(img_dims,att_dims,bias=True)
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0).cuda(),requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10).cuda(),requires_grad=True)
        self.att_dims = att_dims

    def forward(self, AttM, input, label=None, TrainOrTest='Train',clsGroup=None,showAtt=False):
        batch_num = len(input)
        cls_num = len(AttM)
        x = F.normalize(input)
        att = self.img_guaid(x)
        att = self.ReLU(att)
        att = F.softmax(att/self.tmp).reshape(-1,1,self.att_dims)
        att = att + torch.ones_like(att)
        att = att.repeat(1,cls_num,1)

        AttM = AttM.reshape(1,-1,self.att_dims)
        AttM = AttM.repeat(len(input),1,1)

        AttM = att*AttM
         
        W1 = self.L1(AttM)
        W1 = self.ReLU1(W1)
        W2 = self.L2(W1)
        classifier = self.ReLU2(W2)
        classifier = F.normalize(classifier,p=2,dim=-1,eps=1e-12)
        
        out = self.scale_cls*(torch.matmul(classifier,x.t())+self.bias)
        out = out.permute(1,0,2)
        out = torch.diagonal(out,offset=0,dim1=1,dim2=2)
        out = out.t()

        if TrainOrTest=='Train':
            if showAtt:
                return out,att
            else:
                return out
        else:
            if showAtt:
                return out,out/self.scale_cls,att
            else:
                return out,out/self.scale_cls

