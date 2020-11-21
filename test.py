"""
extra the last layer embedding of inception3

refer: https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""

import argparse
from tqdm import tqdm
import numpy as np
import json 
import datetime
import sys
import os
from scipy.io import loadmat
import pickle

import torch.nn.functional as F
import torch.nn as nn
import torch
import yaml
import torchvision.transforms as transforms
import networkx as nx
#import torchvision.datasets as datasets
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import utils
from models.S2ILinearModel import S2ILayer2Model_IAS
TMP=10

class train(object):
    def __init__(self,device,cfg_file,IAS=False):
        self.IAS = IAS
        self.device = device
        with open(cfg_file,'r') as f:
           self.cfg = yaml.safe_load(f) 
        utils.init_seeds(self.cfg['model_hyp']['random_seed'])

        attSplit = loadmat('./dataset/xlsa/'+self.cfg['dataset']['name']+'/att_splits.mat') 
        res101 = loadmat('./dataset/xlsa/'+self.cfg['dataset']['name']+'/res101.mat')
        labels = res101['labels'].astype(int).squeeze() - 1
        seen_dataLoc = attSplit['test_seen_loc'].squeeze() - 1
        unseen_dataLoc = attSplit['test_unseen_loc'].squeeze() -1
        
        seen_labels = labels[seen_dataLoc]
        unseen_labels = labels[unseen_dataLoc]
        self.seen_labels = np.unique(seen_labels)
        self.unseen_labels = np.unique(unseen_labels)

        self.clsname = [ attSplit['allclasses_names'][i][0][0] for i in range(len(attSplit['allclasses_names']))]
        att_matrix = np.transpose(attSplit['att'])
        self.cfg['model_hyp']['att_feats'] = att_matrix.shape[1]

        self.attMatrix = att_matrix.copy()
        self.attMatrix[:len(self.seen_labels)] = att_matrix[self.seen_labels]
        self.attMatrix[len(self.seen_labels):] = att_matrix[self.unseen_labels]

        self.attMatrix = torch.FloatTensor(self.attMatrix).to(self.device)

    def createLinearModel(self,):

        self.model1 = S2ILayer2Model_IAS(att_dims=self.cfg['model_hyp']['att_feats'],img_dims=self.cfg['model_hyp']['img_feats'],tmp=5).to(self.device) 

    def apply_classification_weights(self,features, cls_weights,norm=False):
    
        features = F.normalize(features,dim=-1)
        cls_weights = F.normalize(cls_weights, p=2, dim=-1, eps=1e-12) 
    
        cls_scores = self.scale_cls * (torch.matmul(cls_weights,features.t()))
        cls_scores = cls_scores.permute(0,2,1)
        cls_scores = torch.diagonal(cls_scores,offset=0,dim1=0,dim2=1)
        cls_scores = cls_scores.t()
        if norm:
            return cls_scores,cls_scores/scale_cls
        else:
                return cls_scores

    def IASatt(self,features,AttM,tmp):
        cls_num = len(AttM)
        attdims = AttM.shape[1]
        atten = torch.mm(features,self.w_IAS)+self.b_IAS
        atten = F.relu(atten)
        atten = F.softmax(atten/tmp,dim=1).reshape(-1,1,attdims)
        atten = atten + torch.ones_like(atten)
        atten = atten.repeat(1,cls_num,1)
        AttM = AttM.unsqueeze(0)
        AttM = AttM.repeat(len(features),1,1)
        AttM = atten*AttM
        return AttM
    
    def forward(self,att,features,tmp):
            
        features = features.squeeze()
        att = self.IASatt(features,att,tmp)

        a1 = F.relu(torch.matmul(att, self.w1) + self.b1)
        a2 = F.relu(torch.matmul(a1, self.w2) + self.b2)
        return a2

    def loadDataFromCustom(self,dataType):
        """
            Load Data from Xlsa:
            ------

        """
        print('load from customed')

        with open('./dataset/finetune/'+self.cfg['dataset']['name']+'/dvbeExtracted.pkl','rb') as f:
            ExtractFeature = pickle.load(f)
        imgfs = np.array(ExtractFeature[dataType]['features'])
        labels = np.array(ExtractFeature[dataType]['labels'])

        print(imgfs.shape)
        print(labels.shape)
        select_id, idx = np.unique(labels,return_inverse=True)
        if dataType.lower() == "test_unseen":
            labels = idx + len(self.seen_labels)

        else:
            labels = idx

        imgfs = torch.FloatTensor(imgfs.reshape(-1,2048))
        labels = torch.LongTensor(labels)
        dataset = torch.utils.data.TensorDataset(imgfs,labels)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=self.cfg['dataset_hyp']['batch_size'],shuffle=False)
        return dataloader

    def loadChk(self,chkFile1,chkFile2):
        """
            if resume training
            args:
                chkFile
            return:
                bestAcc in chkFile
        """

        print("=> load checkpoint '{}'".format(chkFile1))
        chk = torch.load(chkFile1)
        self.cfg['model_hyp']['start_epoch'] = chk['epoch']
        print(chk['model'].keys())
        self.model1.load_state_dict(chk['model'])

        print("=> load checkpoint '{}'".format(chkFile2))
        
        model = torch.load(chkFile2)
        self.w1 = Variable(model['w1'], requires_grad=False)
        self.b1 = Variable(model['b1'], requires_grad=False)
        self.w2 = Variable(model['w2'], requires_grad=False)
        self.b2 = Variable(model['b2'], requires_grad=False)
        self.w_IAS = Variable(model['w_IAS'],requires_grad=False)
        self.b_IAS = Variable(model['b_IAS'],requires_grad=False)
        self.scale_cls = Variable(model['scale_cls'], requires_grad=False)
        self.bias = Variable(model['bias'], requires_grad=False)
        return 

    def evaluationSplit(self,dataloader,dataType='seen',accType='allCls',imgfType='customed',recordFile=None,seenOrunseenCls=None):
        """
        evaluation model on input dataloader
        args:
            dataloader
            dataType: 'seen'   : count acc, CEloss, MSEloss 
                      'unseen' : count acc, CEloss(1), MSEloss 
            accType:  'allCls' : return acc mean of all cls 
                      'allImg' : return acc mean of all images
            recordFile: None   : do not record
                        'print'： print eval
                       filepath: record in this file
        return:
            accs: (int) mean of all cls or all images 
                        decide by accType
            accs_cls: (dict) mean acc according to cls
            generalAccs_cls: (dict) mean generalAccs 2 cls,
            splitacc: datasplit bw seen and unseen
            cos_cls:(dict){cls:[sample_cos]} sample cosimilarity 
        """ 

        total_spp_loss = []
        generalAccs = []
        accs = []
        generalAccs_cls = {}
        accs_cls = {}
        cos_cls = {}
        spp_loss_cls = {}
        splitaccs = []
        TF_l = []
        cos = []

        with torch.no_grad():
            for images,targets in tqdm(dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                if self.cfg['dataset_hyp']['imgfType'] == 'customed':
                    images = self.caffeRes101(images).squeeze()

                batch_visual_norm = F.normalize(images, p=2, dim=images.dim()-1, eps=1e-12)                
                generalpreds1,generalpreds1_n = self.model1(self.attMatrix,images,TrainOrTest="Test")
                batch_weights = self.forward(self.attMatrix[len(self.seen_labels):],batch_visual_norm,tmp=TMP)       
                all_cls_weights = batch_weights
                generalpreds2 = self.apply_classification_weights(batch_visual_norm, all_cls_weights)
                generalpreds1 = torch.argmax(generalpreds1,dim=1)
                generalpreds2 = torch.argmax(generalpreds2,dim=1)+len(self.seen_labels)
                generalpreds = generalpreds1
                for i in range(len(generalpreds1)):
                    if generalpreds1[i] > len(self.seen_labels)-1: # judge by model as unseen class
                        generalpreds[i] = generalpreds2[i]
               
                if dataType.lower()=='seen':
                    preds,_ = self.model1(self.attMatrix[:len(self.seen_labels)],images,TrainOrTest="Test")
                    preds = torch.argmax(preds,dim=1)
                elif dataType.lower()=='unseen':
                    preds = generalpreds2

                # seen or unseen split accuracy
                splitacc = [ i in seenOrunseenCls for i in generalpreds1.cpu().detach().tolist()]
                splitaccs.extend(splitacc)

                # calculate accs
                acc = preds == targets
                TF_l.extend(acc.cpu().detach().tolist())
                max_cos,_ = torch.max(generalpreds1_n,dim=-1)
                cos.extend(max_cos.cpu().detach().tolist())

                # calculate accs_cls
                for c in torch.unique(targets):
                    loc = targets == c
                    acc_c = preds[loc] == targets[loc]
                    generalacc_c = generalpreds[loc] == targets[loc]
                    dist_c = generalpreds1_n[loc,c].cpu().detach().tolist()
                    acc_c = acc_c.cpu().detach().tolist()
                    generalacc_c = generalacc_c.cpu().detach().tolist()
                    try:
                        accs_cls[c.cpu().detach().tolist()].extend(acc_c)
                        generalAccs_cls[c.cpu().detach().tolist()].extend(generalacc_c)
                        cos_cls[c.cpu().detach().tolist()].extend(dist_c)
                    except(KeyError):
                        accs_cls[c.cpu().detach().tolist()] = acc_c
                        generalAccs_cls[c.cpu().detach().tolist()] = generalacc_c
                        cos_cls[c.cpu().detach().tolist()] = dist_c

        # average class accuracy
        if accType == 'allCls':
            for c in accs_cls:
                accs.append(np.mean(accs_cls[c]))
                generalAccs.append(np.mean(generalAccs_cls[c]))
        for c in accs_cls:
            accs_cls[c] = np.mean(accs_cls[c])*100
            generalAccs_cls[c] = np.mean(generalAccs_cls[c])*100

        return  {'accs':np.mean(accs)*100,
                 'generalAccs':np.mean(generalAccs)*100,
                 'generalAccs_cls':generalAccs_cls,
                 'accs_cls':accs_cls,
                 'splitacc':np.mean(splitaccs)*100,
                 'cos_cls':cos_cls,
                 'TrueFalseList':TF_l,
                 'cos':cos
                }

    def prepareModel(self,):
        self.createLinearModel()

    def evalModel(self,chkFile1,chkFile2,resume=False):
        """
            train Model #epochs defined in cfg

            args:
                chkFile: save model chkpoint
        """

        testSeenData = self.loadDataFromCustom(dataType='test_seen')
        testUnseenData = self.loadDataFromCustom(dataType='test_unseen')

        self.loadChk(chkFile1,chkFile2)
        
        testSeenEval = self.evaluationSplit(testSeenData,dataType='seen',accType=self.cfg['model_hyp']['accType'],imgfType=self.cfg['dataset_hyp']['imgfType'],seenOrunseenCls=np.arange(0,len(self.seen_labels)))
        testUnseenEval = self.evaluationSplit(testUnseenData,dataType='unseen',accType=self.cfg['model_hyp']['accType'],imgfType=self.cfg['dataset_hyp']['imgfType'],seenOrunseenCls=np.arange(len(self.seen_labels),len(self.seen_labels)+len(self.unseen_labels)))

        accu = testUnseenEval['generalAccs']
        accs = testSeenEval['generalAccs']
        H = 2*(accu*accs)/(accu+accs)
        print('test/UnseenAcc',testUnseenEval['accs'])
        print('test/SeenAcc',testSeenEval['accs'])
        print('test/H',H)
        print('test/generalAccs',accs)
        print('test/generalAccu',accu)
        print('test/fu',testUnseenEval['splitacc'])
        print('test/fs',testSeenEval['splitacc'])
        return testSeenEval['TrueFalseList'],testSeenEval['cos']

if __name__ =="__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument('--modelChk',type=str,help='model chkpoint file')
    parser.add_argument('--yaml',type=str,help='checkpoint file saved in yaml')
    parser.add_argument('--device',default='cuda:0',type=str,help='cuda:0,cuda:1,cuda:2,cpu'
)
    opt = parser.parse_args()

    #chkFile1 ='./chk/201115CUBS2IFinetune_ResNet101_fullv_r520_IAStmp10CustomedExtractedCos520LR5e-4WD5e-7/chk_best_Hs.pt' #class weight + IAS
    chkFile1 ='./chk/CUBLR5e-4WD9e-7/chk_best_Hs.pt' #class weight + IAS
    chkFile2 = './chk/cublr5e-4_opt5e-7_w20_s4_51FinetunebestunseenAcc.pt'#mixup_IAS
    device = torch.device(opt.device) 

    t = train(device,opt.yaml,IAS=True)
    t.prepareModel()

    truefalselistBase,cos_Base = t.evalModel(chkFile1,chkFile2)
    


