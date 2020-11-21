"""
Part of this code is from Kai Li "kailigo". The website is https://github.com/kailigo/cvcZSL.
"""
from scipy import io
import numpy as np
import torch
from pdb import set_trace as breakpoint
import torch.utils.data as data


class data_loader_virtualCls(data.Dataset):
        def __init__(self, feats, atts, labels,  ways=16, shots=4):
                self.ways = ways*2                
                self.shots = shots    

                self.feats = torch.tensor(feats).float()
                self.atts = torch.tensor(atts).float()
                self.labels = labels
                self.classes = np.unique(labels)
                
        def __getitem__(self, index):
                is_first = True
                select_feats = []
                select_atts = []
                select_labels = []        
                select_labels = torch.LongTensor(self.ways*self.shots)
                selected_classes = np.random.choice(list(self.classes), self.ways, False)
                #selected_classes = self.classes
                #mixup
                cls_idx = {}


                for i in range(len(selected_classes)):
                        idx = (self.labels==selected_classes[i]).nonzero()[0]
                        select_instances = np.random.choice(idx, self.shots, False)
                        lam = np.random.beta(5, 1)

                        #if selected_classes[i] in [2,15]:
                        #    cls_idx[selected_classes[i]] = i
                        #lam = np.random.uniform(0.49,0.51)

                        if i<self.ways/2:
                            for j in range(self.shots):
                                    feat = self.feats[select_instances[j], :]
                                    att = self.atts[select_instances[j], :]  

                                    feat = feat.unsqueeze(0)
                                    att = att.unsqueeze(0)
                                    # print(feat.size())
                                    # print(att.size())
                                    if is_first:
                                            is_first=False
                                            select_feats = feat
                                            select_atts = att
                                    else:                   
                                            select_feats = torch.cat((select_feats, feat),0)                
                                            select_atts = torch.cat((select_atts, att),0)                
                                    select_labels[i*self.shots+j] = i
                        else:
                            for j in range(self.shots):
                                    feat = self.feats[select_instances[j], :]
                                    att = self.atts[select_instances[j], :]  

                                    feat = feat.unsqueeze(0)
                                    feat = lam*feat+(1-lam)*select_feats[int(i-self.ways/2)*self.shots+j]
                                    att = att.unsqueeze(0)
                                    att = lam*att+(1-lam)*select_atts[int(i-self.ways/2)*self.shots+j]
                                    # print(feat.size())
                                    # print(att.size())
                                    select_feats = torch.cat((select_feats, feat),0)                
                                    select_atts = torch.cat((select_atts, att),0)                
                                    select_labels[i*self.shots+j] = i - int(self.ways/2)

                noval_index = int(self.ways/2)*self.shots

                return select_feats[noval_index:], select_atts[noval_index:], select_labels[noval_index:]
                
        def __len__(self):
                return self.__size

