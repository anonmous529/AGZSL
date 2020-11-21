#!/bin/bash

data='CUB'
device=2
imgType='CustomedExtracted'

random_seed=520
Lr="5e-4"
WeightDecay="9e-7"

special_=$data'LR'$Lr'WD'$WeightDecay

chkFolder='chk/'$special_
yamlFile=$chkFolder'/seenExpert.yaml'

if [ -d "$chkFolder" ]; then
	echo "$chkFolder has existed"
else
	mkdir $chkFolder
fi

# create yamlFile
python -m mkseenCFG --yamlFile $yamlFile --random_seed $random_seed --lr $Lr --weightDecay $WeightDecay --imgType $imgType --datasetName $data

# train mode

modelChk=$chkFolder'/chk'
recordFile=$chkFolder'/record.txt'
summaryFolder='summary/'$special_

if [ -d "$summaryFolder" ]; then
	echo "$summaryFolder has existed"
	rm -rf $summaryFolder	
	mkdir $summaryFolder

else
	
	mkdir $summaryFolder
fi

CUDA_VISIBLE_DEVICES=$device python -m train_seen --train --modelChk $modelChk --recordFile $recordFile --yaml $yamlFile --summaryFolder $summaryFolder
	
	
