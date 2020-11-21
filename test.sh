#!/bin/bash


# create Chk Folder 
data='CUB'
device=0
dir='S2I'
ClassifierType='Cos' # Cos 
imgType='CustomedExtracted'
trainingCls='Seen'
random_seed=301


		
chkFolder='./chk/CUBLR5e-4WD9e-7'   
#chkFolder='/home/tim/project/ZSL/exp/200825Iguaid/cub/chkCUBXLSASeenLRWD/200904CUBS2IImgfCaffeResNet101_fullv_clsweight_inital_XLSASeenCos301LR5e-3WD4e-7'   
yamlFile=$chkFolder'/seenExpert.yaml'

CUDA_VISIBLE_DEVICES=$device python -m test --yaml $yamlFile #--modelChk $modelChk
	
	
