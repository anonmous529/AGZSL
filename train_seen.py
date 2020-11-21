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
import pretrainedmodels
from PIL import Image


from tensorboardX import SummaryWriter
from models.S2ILinearModel import S2ILayer2Model_IAS
import utils 

class ToSpaceBGR(object):
    def __init__(self,is_bgr):
        self.is_bgr = is_bgr
    def __call__(self,tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor

class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255
    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor

class train(object):
    def __init__(self,device,cfg_file,summaryFolder):
        self.device = device
        self.summaryFolder = summaryFolder
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
        pass


    def mkSummary(self,summaryFolder):
        self.writer = SummaryWriter(summaryFolder) 

    def createLinearModel(self,):
        """
           build Linear Model use attribute build Adjacency matrix  
        """ 

        self.model = S2ILayer2Model_IAS(att_dims=self.cfg['model_hyp']['att_feats'],img_dims=self.cfg['model_hyp']['img_feats'],tmp=10).to(self.device) 

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

        select_id, idx = np.unique(labels,return_inverse=True)
        if dataType.lower() == "test_unseen":
            labels = idx + len(self.seen_labels)

        else:
            labels = idx

        imgfs = torch.FloatTensor(imgfs.reshape(-1,2048))
        labels = torch.LongTensor(labels)

        dataset = torch.utils.data.TensorDataset(imgfs,labels)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=self.cfg['dataset_hyp']['batch_size'],shuffle=True)
        return dataloader

    def loss(self,):
        """
            loss function for Linear model
            output is Cosine Similarity
        """
        self.lossf = nn.CrossEntropyLoss(weight=None).to(self.device)

    def optim(self,):
        """
            optimizer for Attribute Adejacenciy GCN
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(),self.cfg['model_hyp']['lr'],weight_decay=self.cfg['model_hyp']['weight_decay'])

    
    def scheduler(self,):
        """
            scheduler for optimizer
        """
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.cfg['model_hyp']['epochs'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=2,gamma=0.87)

    def loadChk(self,chkFile):
        """
            if resume training
            args:
                chkFile
            return:
                bestAcc in chkFile
        """

        print("=> load checkpoint '{}'".format(chkFile))
        chk = torch.load(chkFile)
        self.cfg['model_hyp']['start_epoch'] = chk['epoch']
        #best_acc = chk['best_acc']
        self.model.load_state_dict(chk['model'])
        self.optimizer.load_state_dict(chk['optimizer'])
        self.scheduler.load_state_dict(chk['scheduler'])
        
        #return best_acc

    def saveModel(self,epoch,Info,model,optimizer,scheduler,trainSeenEval,testSeenEval,testUnseenEval,H,chkfile):
        """
            save Model
        """
        chk = {} 
        chk['model'] = model.state_dict() 
        chk['optimizer'] = optimizer.state_dict()
        chk['scheduler'] = scheduler.state_dict()
        chk['Info'] = Info          #'test/UnseenAcc:%.2f test/SeenAcc:%.2f H:%.2f \nfs:%.2f fu:%.2f'
        chk['eval'] = {'trainSeen':trainSeenEval,\
                       'testSeen': testSeenEval,\
                       'testUnseen':testUnseenEval,\
                       'H':H} 

        chk['epoch'] = epoch

        torch.save(chk, chkfile)
        

    def trainModel_1Epoch(self,dataloader,imgfType='customed',recordFile=None):
        
        self.model.train()
        self.scheduler.step()
        
        for images,targets in tqdm(dataloader):

            images = images.to(self.device)
            targets = targets.to(self.device)

            if self.cfg['dataset_hyp']['imgfType'] == 'customed':
                images = self.caffeRes101(images).squeeze()

            preds = self.model(self.attMatrix[:len(self.seen_labels)],images,targets,TrainOrTest="Train")


            # use which loss function
            loss = self.lossf(preds,targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1)
            self.optimizer.step()

            loss = loss.detach().cpu().tolist()

        # record
        if recordFile:
            with open(recordFile,'a')as f:
                f.write('{:>17.3e}'.format(loss))

        return  {
                     'loss':np.mean(loss),
                }

    def evaluation(self,dataloader,dataType='seen',accType='allCls',imgfType='customed',recordFile=None,seenOrunseenCls=None):
        """
        evaluation model on input dataloader
        args:
            dataloader
            dataType: 'seen'   : count acc, CEloss, MSEloss 
                      'unseen' : count acc, CEloss(1), MSEloss 
            accType:  'allCls' : return acc mean of all cls 
                      'allImg' : return acc mean of all images
            imgfType: 'customed': use resnet101 features
                       'xlsa'   : xlsa resnet101 features
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

        # imgCls generated from semantic adjacency matrix GCN

        total_spp_loss = []
        generalAccs = []
        accs = []
        generalAccs_cls = {}
        accs_cls = {}
        cos_cls = {}
        spp_loss_cls = {}
        splitaccs = []
        splitacc_cls = {}

        with torch.no_grad():
            for images,targets in tqdm(dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                if self.cfg['dataset_hyp']['imgfType'] == 'customed':
                    images = self.caffeRes101(images).squeeze()

                generalpreds,generalpreds1 = self.model(self.attMatrix,images,TrainOrTest="Test")

                if dataType.lower()=='seen':
                    preds,_ = self.model(self.attMatrix[:len(self.seen_labels)],images,TrainOrTest="Test")
                    preds = torch.argmax(preds,dim=1)
                elif dataType.lower()=='unseen':
                    preds,_ = self.model(self.attMatrix[len(self.seen_labels):],images,TrainOrTest="Test")
                    preds = torch.argmax(preds,dim=1)+len(self.seen_labels)

                generalpreds = torch.argmax(generalpreds,dim=1)

                # seen or unseen split accuracy
                splitacc = [ i in seenOrunseenCls for i in generalpreds.cpu().detach().tolist()]
                splitaccs.extend(splitacc)
                splitacc = np.array(splitacc)
                

                # calculate accs
                if accType == 'allImages':
                    acc = preds == targets
                    accs.extend(acc.cpu().detach().tolist())

                # calculate accs_cls
                for c in torch.unique(targets):
                    loc = targets == c
                    acc_c = preds[loc] == targets[loc]
                    generalacc_c = generalpreds[loc] == targets[loc]

                    
                    dist_c = generalpreds1[loc,c].cpu().detach().tolist()
                    acc_c = acc_c.cpu().detach().tolist()
                    generalacc_c = generalacc_c.cpu().detach().tolist()
                    try:
                        accs_cls[c.cpu().detach().tolist()].extend(acc_c)
                        generalAccs_cls[c.cpu().detach().tolist()].extend(generalacc_c)
                        cos_cls[c.cpu().detach().tolist()].extend(dist_c)
                        splitacc_cls[c.cpu().detach().tolist()].extend(list(splitacc[loc.cpu().detach().numpy()]))
                    except(KeyError):
                        accs_cls[c.cpu().detach().tolist()] = acc_c
                        generalAccs_cls[c.cpu().detach().tolist()] = generalacc_c
                        cos_cls[c.cpu().detach().tolist()] = dist_c
                        splitacc_cls[c.cpu().detach().tolist()] = list(splitacc[loc.cpu().detach().numpy()])

        # average class accuracy
        if accType == 'allCls':
            for c in accs_cls:
                accs.append(np.mean(accs_cls[c]))
                generalAccs.append(np.mean(generalAccs_cls[c]))
        for c in accs_cls:
            accs_cls[c] = np.mean(accs_cls[c])*100
            generalAccs_cls[c] = np.mean(generalAccs_cls[c])*100
            splitacc_cls[c] = np.mean(splitacc_cls[c])*100

        return  {'accs':np.mean(accs)*100,
                 'generalAccs':np.mean(generalAccs)*100,
                 'generalAcc_cls':generalAccs_cls,
                 'acc_cls':accs_cls,
                 'splitacc':np.mean(splitaccs)*100,
                 'splitacc_cls':splitacc_cls,
                 'cos_cls':cos_cls
                }

    def prepareModel(self,):
        self.createLinearModel()
        self.mkSummary(self.summaryFolder)
        self.loss()
        self.optim()
        self.scheduler()

    def trainModel(self,chkFile,recordFile,resume=False):
        """
            train Model #epochs defined in cfg

            args:
                chkFile: save model chkpoint
                recordFile: save acc and loss
        """

        trainData = self.loadDataFromCustom(dataType='train_seen')
        testSeenData = self.loadDataFromCustom(dataType='test_seen')
        testUnseenData = self.loadDataFromCustom(dataType='test_unseen')
        bestHs = 0
        
        for epoch in range(self.cfg['model_hyp']['start_epoch'],self.cfg['model_hyp']['epochs']):
            print('modelhyp:',chkFile.strip('/chk'),'epoch:',epoch)
            with open(recordFile,'a') as f:
                f.write('{:^5d}'.format(epoch))

            # train S2I
            trainSeenEval=self.trainModel_1Epoch(trainData,imgfType=self.cfg['dataset_hyp']['imgfType'])

            testSeenEval = self.evaluation(testSeenData,dataType='seen',accType=self.cfg['model_hyp']['accType'],recordFile=recordFile,imgfType=self.cfg['dataset_hyp']['imgfType'],seenOrunseenCls=np.arange(0,len(self.seen_labels)))
            testUnseenEval = self.evaluation(testUnseenData,dataType='unseen',accType=self.cfg['model_hyp']['accType'],recordFile=recordFile,imgfType=self.cfg['dataset_hyp']['imgfType'],seenOrunseenCls=np.arange(len(self.seen_labels),len(self.seen_labels)+len(self.unseen_labels)))

            #testSeenEval = self.evaluation(testSeenData,dataType='seen',accType=self.cfg['model_hyp']['accType'],recordFile=recordFile)
            #testUnseenEval = self.evaluation(testUnseenData,dataType='unseen',accType=self.cfg['model_hyp']['accType'],recordFile=recordFile) 
            seenCos = np.mean([ np.mean(testSeenEval['cos_cls'][i]) for i in testSeenEval['cos_cls']])
            unseenCos = np.mean([ np.mean(testUnseenEval['cos_cls'][i]) for i in testUnseenEval['cos_cls']])
            difseenCosBWunseenCos = seenCos - unseenCos

            generalAccs = testSeenEval['generalAccs']
            generalAccu = testUnseenEval['generalAccs']
            H = 2*generalAccs*generalAccu/(generalAccs+generalAccu)

            accs = testSeenEval['accs']
            accu = testUnseenEval['accs']
            fs = testSeenEval['splitacc']
            fu = testUnseenEval['splitacc']
            Hs = 2*fs*fu/(fs+fu)

            self.writer.add_scalar('general/generalSeenAcc',generalAccs,epoch)
            self.writer.add_scalar('general/genealUnseenAcc',generalAccu,epoch)
            self.writer.add_scalar('general/H',H,epoch)
            self.writer.add_scalar('split/SeenAcc',accs,epoch)
            self.writer.add_scalar('split/UnseenAcc',accu,epoch)

            self.writer.add_scalar('loss/loss',trainSeenEval['loss'],epoch)
            self.writer.add_scalar('split/fs',fs,epoch)
            self.writer.add_scalar('split/fu',fu,epoch)
            self.writer.add_scalar('split/Hs',Hs,epoch)
            self.writer.add_scalar('split/seenCos',seenCos,epoch)
            self.writer.add_scalar('split/unseenCos',unseenCos,epoch)
            self.writer.add_scalar('split/seenCos-unseenCos',difseenCosBWunseenCos,epoch)


            for i in range(len(self.seen_labels),len(self.seen_labels)+len(self.unseen_labels)):
                self.writer.add_scalar('SplitUnseen/'+str(i)+self.clsname[self.unseen_labels[i-len(self.seen_labels)]],testUnseenEval['splitacc_cls'][i],epoch)
                self.writer.add_scalar('UnseenAccs/'+str(i)+self.clsname[self.unseen_labels[i-len(self.seen_labels)]],testUnseenEval['acc_cls'][i],epoch)
                self.writer.add_scalar('generalUnseen/'+str(i)+self.clsname[self.unseen_labels[i-len(self.seen_labels)]],testUnseenEval['generalAcc_cls'][i],epoch)
            for i in range(len(self.seen_labels)):
                self.writer.add_scalar('SplitSeen/'+str(i)+self.clsname[self.seen_labels[i]],testSeenEval['splitacc_cls'][i],epoch)
                self.writer.add_scalar('SeenAccs/'+str(i)+self.clsname[self.seen_labels[i]],testSeenEval['acc_cls'][i],epoch)
                self.writer.add_scalar('generalseen/'+str(i)+self.clsname[self.seen_labels[i]],testSeenEval['generalAcc_cls'][i],epoch)

            Info ='generalAccs:%.2f generalAccu:%.2f H:%.2f \n Accs:%.2f Accu:%.2f fs:%.2f fu:%.2f Hs:%.2f'%(generalAccs,generalAccu,H,accs,accu,testSeenEval['splitacc'],testUnseenEval['splitacc'],Hs)
            print(Info)

            with open(recordFile,'a') as f:
                f.write('\n')


            if Hs > bestHs:
                bestHs = Hs
                self.saveModel(epoch,Info,self.model,self.optimizer,self.scheduler,trainSeenEval,testSeenEval,testUnseenEval,Hs,chkFile+'_best_Hs.pt')
                best_HsInfo = Info

            if epoch%500 == 0 and epoch > 0: 
                self.saveModel(self.cfg['model_hyp']['start_epoch'],Info,self.model,self.optimizer,self.scheduler,trainSeenEval,testSeenEval,testUnseenEval,H,chkFile+'_'+str(epoch)+'.pt')

        print('bestHs')
        print(best_HsInfo)

if __name__ =="__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument('--modelChk',type=str,help='model chkpoint file')
    parser.add_argument('--recordFile',type=str,help='record loss and acc file')
    parser.add_argument('--summaryFolder',type=str,help='tensorboardX file folder')
    parser.add_argument('--yaml',type=str,help='checkpoint file saved in yaml')
    parser.add_argument('--device',default='cuda:0',type=str,help='cuda:0,cuda:1,cuda:2,cpu'
)
    parser.add_argument('--train',action='store_true',help='train or just eval')
    parser.add_argument('--resume',action='store_true',help='resume to train model')
    opt = parser.parse_args()

    chkFile = opt.modelChk
    recordFile = opt.recordFile
    device = torch.device(opt.device) 

    t = train(device,opt.yaml,opt.summaryFolder)
    t.prepareModel()

    if opt.train:
        # train model
        t.trainModel(chkFile=chkFile,recordFile=recordFile,resume=opt.resume)
    else:
        # evaluate model
        t.evalModel(chkFile)

