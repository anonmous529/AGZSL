"""
Part of code is from Kai Li "kailigo". The gitub website is https://github.com/kailigo/cvcZSL.
We add virtual classes and IAS in the code.
"""
from scipy import io
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import lr_scheduler
import torch.utils.data as data
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter

from utils import ReDirectSTD
from unseen_data_loader import data_loader_virtualCls
from unseen_option import Options

import os
import random
import pickle
# from test_embeded import test_while_training_simple

TMP = 10

args = Options().parse()
model_file_name = './chk/' + args.model_file
summaryFolder = './summary/' + args.log_file
if not os.path.exists(summaryFolder):
    os.mkdir(summaryFolder)
writer = SummaryWriter(summaryFolder)
print(args)

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    # torch cuda
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

init_seeds(520)
def calc_accuracy(test_visual, test_label, attM, test_id, test_id_seen_unseen,cossim=False):       
        outpred = [0] * test_visual.shape[0]    
        end = 0
        outpred_list = []
        zslpred_list = []
        score_n_list = []
        for j in range(0,len(test_visual),64):
            if j+64>len(test_visual):
                end = len(test_visual)
            else:
                end = j+64
            all_cls_weights = forward(attM,test_visual[j:end],tmp=TMP)       
            score,score_n=apply_classification_weights(test_visual[j:end].cuda(), 
                    all_cls_weights,norm=True)
            score = score.squeeze(0)
            score_n = score_n.squeeze(0)
            
            _, pred = score.max(dim=1)
            pred = pred.view(-1)
            select_test_label = test_label[j:end].view(-1)

            outpred_list.extend(test_id[pred.cpu().detach().numpy()])
            zslpred_list.extend([ i in test_id_seen_unseen for i in test_id[pred.cpu().detach().numpy()]])
            score_n_list.extend(score_n.cpu().detach().numpy())

        seen_unseen_acc = accuracy_score(np.ones(len(zslpred_list)),zslpred_list)
        
        outpred = np.array(outpred_list, dtype='int')
        score_n = np.array(score_n_list)
        test_label = test_label.numpy()
        unique_labels = np.unique(test_label)
        acc = 0
        acc_cls = {}
        preds_cls = {}
        cos_cls = {}
        for l in unique_labels:
                idx = np.nonzero(test_label == l)[0]
                acc_cls[l] = accuracy_score(test_label[idx],outpred[idx])
                try:
                    preds_cls[l].extend(list(outpred[idx])) 
                except(KeyError):
                    preds_cls[l] = list(outpred[idx])

                acc += acc_cls[l]
                loc = test_label == l
                outpred_l = outpred[idx]
                score_n_l = score_n[idx]
                loc = outpred_l == l

                if cossim:
                    cos_c = score_n_l[loc,l]
                    try:
                        cos_cls[l].extend(cos_c)
                    except(KeyError):
                        cos_cls[l] = [cos_c]
        acc = acc / unique_labels.shape[0]

        return acc,seen_unseen_acc,acc_cls,cos_cls,preds_cls


def compute_accuracy_all(test_att, att_all, test_visual_unseen, test_id_unseen, test_label_unseen,
                test_visual_seen, test_id_all, test_label_seen,train_id):

        acc_zsl,_,unseenacc_cls,_,unseenpred_cls = calc_accuracy(test_visual_unseen, test_label_unseen, test_att, test_id_unseen,test_id_unseen)
        acc_seenAcc,_,seenacc_cls,_,_ = calc_accuracy(test_visual_seen, test_label_seen, att_all, train_id, train_id)

        att_all_cls = torch.cat((att_all, test_att))
        
        acc_gzsl_unseen,Ru,unseengeneralAcc_cls,unseenCos_cls,_ = calc_accuracy(test_visual_unseen, test_label_unseen, att_all_cls, test_id_all,test_id_unseen,cossim=True)
        
        acc_gzsl_seen,Rs,seengeneralAcc_cls,seenCos_cls,_ = calc_accuracy(test_visual_seen, test_label_seen, att_all_cls, test_id_all,train_id,cossim=True)       
        acc_cls = {**unseenacc_cls,**seenacc_cls}
        generalAcc_cls = {**unseengeneralAcc_cls,**seengeneralAcc_cls}
        H = 2 * acc_gzsl_seen * acc_gzsl_unseen / (acc_gzsl_seen + acc_gzsl_unseen)

        return acc_zsl, acc_seenAcc, acc_gzsl_unseen, acc_gzsl_seen, H, Rs, Ru, acc_cls, generalAcc_cls,unseenCos_cls,seenCos_cls,unseenpred_cls


def apply_classification_weights(features, cls_weights,norm=False):

        features = F.normalize(features,dim=-1)
        cls_weights = F.normalize(cls_weights, p=2, dim=-1, eps=1e-12) 

        cls_scores = scale_cls * (torch.matmul(cls_weights,features.t()))
        cls_scores = cls_scores.permute(0,2,1)
        cls_scores = torch.diagonal(cls_scores,offset=0,dim1=0,dim2=1)
        cls_scores = cls_scores.t()
        if norm:
            return cls_scores,cls_scores/scale_cls
        else:
            return cls_scores


def IASatt(features,AttM,tmp):
        cls_num = len(AttM)
        attdims = AttM.shape[1]
        atten = torch.mm(features,w_IAS)+b_IAS
        atten = F.softmax(atten/tmp,dim=1).reshape(-1,1,attdims)
        atten = atten.reshape(-1,1,attdims)
        atten = atten + torch.ones_like(atten)
        atten = atten.repeat(1,cls_num,1)
        AttM = AttM.unsqueeze(0)
        AttM = AttM.repeat(len(features),1,1)
        AttM = atten*AttM
        return AttM

def forward(att,features,tmp):
        features = features.squeeze()
        att = IASatt(features,att,tmp)

        a1 = F.relu(torch.matmul(att, w1) + b1)
        a2 = F.relu(torch.matmul(a1, w2) + b2)

        return a2

dataroot = './dataset/xlsa/'
image_embedding = 'res101' 
class_embedding = 'att'
dataset = args.dataset
matcontent = io.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")

feature = matcontent['features'].T
label = matcontent['labels'].astype(int).squeeze() - 1
matcontent = io.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")

trainvalloc = matcontent['trainval_loc'].squeeze() - 1
test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

att_name = 'att'
attribute = matcontent[att_name].T 

chkFile = './dataset/finetune/'+args.dataset+'/dvbeExtracted.pkl'
with open(chkFile,'rb') as f:
    Feature_target = pickle.load(f)

clsname = [ matcontent['allclasses_names'][i][0][0] for i in range(len(matcontent['allclasses_names']))]

train_x = Feature_target['train_seen']['features']
train_label = Feature_target['train_seen']['labels']
train_att = attribute[train_label]
train_id, idx = np.unique(train_label, return_inverse=True)
train_att_unique = attribute[train_id]

test_x_unseen = Feature_target['test_unseen']['features']
test_label_unseen = Feature_target['test_unseen']['labels']
test_id, idx = np.unique(test_label_unseen, return_inverse=True)
att_pro = attribute[test_id]
train_test_id = np.concatenate((train_id, test_id))

test_x_seen = Feature_target['test_seen']['features'] 
test_label_seen = Feature_target['test_seen']['labels']
_, idx = np.unique(test_label_seen, return_inverse=True)

att_dim = train_att.shape[1]
feat_dim = train_x.shape[1]

att_pro = torch.from_numpy(att_pro).float().cuda()
test_x_seen = torch.from_numpy(test_x_seen).float().cuda()
test_x_seen = F.normalize(test_x_seen, p=2, dim=test_x_seen.dim()-1, eps=1e-12)
test_x_unseen = torch.from_numpy(test_x_unseen).float().cuda()
test_x_unseen = F.normalize(test_x_unseen, p=2, dim=test_x_unseen.dim()-1, eps=1e-12)
test_label_seen = torch.tensor(test_label_seen)
test_label_unseen = torch.tensor(test_label_unseen)

att_all = torch.from_numpy(train_att_unique).float().cuda()

bias = nn.Parameter(torch.FloatTensor(1).fill_(0).cuda(), requires_grad=True)
scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10).cuda(), requires_grad=True)
w1 = Variable(torch.FloatTensor(att_dim, args.hidden_dim).cuda(), requires_grad=True)
b1 = Variable(torch.FloatTensor(args.hidden_dim).cuda(), requires_grad=True)
w2 = Variable(torch.FloatTensor(args.hidden_dim, 2048).cuda(), requires_grad=True)
b2 = Variable(torch.FloatTensor(2048).cuda(), requires_grad=True)
w_IAS = Variable(torch.FloatTensor(2048, att_dim).cuda(), requires_grad=True)
b_IAS = Variable(torch.FloatTensor(att_dim).cuda(), requires_grad=True)

w1.data.normal_(0, 0.02)
w2.data.normal_(0, 0.02)
b1.data.fill_(0)
b2.data.fill_(0)
w_IAS.data.normal_(0,0.02)
b_IAS.data.fill_(0)

optimizer = torch.optim.Adam([w_IAS,b_IAS,w1, b1, w2, b2, bias, scale_cls], lr=args.lr, weight_decay=args.opt_decay)

# breakpoint()
step_size = args.step_size
gamma = args.gamma
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
criterion = nn.CrossEntropyLoss()

ways = args.ways
shots = args.shots

dataset = data_loader_virtualCls(train_x, train_att, train_label, ways=ways, shots=shots)

# breakpoint()
best_acc_zsl = 0.0
best_acc_gzsl_seen = 0.0
best_acc_gzsl_unseen = 0.0
best_H = 0.0
best_epoch = 0
best_unseenAcc = 0.0


for epoch in range(args.num_epochs):    
        epoch_loss = 0
        lr_scheduler.step()

        for i in range(1000):           
                batch_visual, batch_att, batch_label = dataset.__getitem__(i)                           
                batch_visual = batch_visual.cuda()                              
                batch_visual_norm = F.normalize(batch_visual, p=2, dim=batch_visual.dim()-1, eps=1e-12)                         

                indx = torch.tensor(list(range(0, ways*shots, shots)))  
                unique_batch_att = torch.index_select(batch_att, 0, indx).float().cuda()                

                batch_weights = forward(unique_batch_att,batch_visual_norm,tmp=TMP)       
                all_cls_weights = batch_weights

                score = apply_classification_weights(batch_visual_norm, all_cls_weights)
                score = score.squeeze(0)                
                loss = criterion(score, Variable(batch_label.cuda()))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([w_IAS,b_IAS,w1, b1, w2, b2, scale_cls, bias], 1)
                optimizer.step()
                epoch_loss = epoch_loss + loss

        epoch_loss = epoch_loss / 1000.0
        epoch_loss = epoch_loss.data.cpu().numpy()

        acc_zsl, seenAcc, acc_unseen_gzsl, acc_seen_gzsl, H, Rs,Ru,accs_cls,generalAccs_cls,unseenCos_cls,seenCos_cls,unseenpred_cls = compute_accuracy_all(att_pro, att_all, test_x_unseen, 
                test_id, test_label_unseen, test_x_seen, train_test_id,  test_label_seen, train_id)
        
        H = 2 * acc_seen_gzsl * acc_unseen_gzsl / (acc_seen_gzsl + acc_unseen_gzsl)
        writer.add_scalar('general/acc_seen_gzsl',acc_seen_gzsl,epoch)
        writer.add_scalar('general/acc_unseen_gzsl',acc_unseen_gzsl,epoch)
        writer.add_scalar('general/H',H,epoch)
        writer.add_scalar('split/unseenAcc',acc_zsl,epoch)
        writer.add_scalar('split/seenAcc',seenAcc,epoch)
        writer.add_scalar('split/Rs',Rs,epoch)
        writer.add_scalar('split/Ru',Ru,epoch)
        writer.add_scalar('loss/loss',epoch_loss,epoch)
        

        if acc_zsl > best_unseenAcc:
                print('save best acc')
                best_unseenAcc = acc_zsl
                best_epoch = epoch
                best_acc_zsl = acc_zsl          
                best_seen_acc = seenAcc          
                best_acc_gzsl_seen = acc_seen_gzsl
                best_acc_gzsl_unseen = acc_unseen_gzsl
                best_H = H
                best_Ru = Ru
                best_Rs = Rs

                best_w1 = w1.data.clone()
                best_b1 = b1.data.clone()
                best_w2 = w2.data.clone()
                best_b2 = b2.data.clone()
                best_scale_cls = scale_cls.data.clone()
                best_bias = bias.data.clone()

                torch.save({'w1': best_w1, 'b1': best_b1, 'w2': best_w2, 'b2': best_b2, 
                    'scale_cls': best_scale_cls, 'bias': best_bias,'w_IAS':w_IAS,'b_IAS':b_IAS}, model_file_name.replace('.pt','bestunseenAcc.pt'))
                

        for param_group in optimizer.param_groups:
                print('ep: %d,  lr: %lf, loss: %.4f,  zsl: %.4f, seenAcc: %.4f  gzsl: seen=%.4f, unseen=%.4f, h=%.4f, Rs=%.4f, Ru=%.4f ' % 
                        (epoch, param_group['lr'],  epoch_loss, acc_zsl, seenAcc, acc_seen_gzsl, acc_unseen_gzsl, H, Rs, Ru,))            
print(model_file_name)
print('best_ep: %d, zsl: %.4f, seenAcc: %.4f  gzsl: seen=%.4f, unseen=%.4f, h=%.4f, Rs=%.4f, Ru=%.4f' % 
        (best_epoch, best_acc_zsl,best_seen_acc, best_acc_gzsl_seen, best_acc_gzsl_unseen, best_H, best_Rs, best_Ru))   

