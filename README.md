## Reference 
We take Kai Li "https://github.com/kailigo/cvcZSL" and "https://github.com/mboboGO/DVBE" as reference. 

## Download finetuned data
We realse fine-tuned images features of CUB on https://drive.google.com/file/d/1G6ZLwANmGqAApVZp3HymBY6dvHAZvROh/view?usp=sharing.
Please download the features and put into folder ./dataset/finetune/CUB/ .
We will realse fine-tuned image features of other dataset in the feature.

## environment setting:
python version: 3.7.6

## The runing commands are below:
### Training unseen expert.
sh train_unseen.sh
### Training seen expert.
sh train_seen.sh
### Evaluation.
sh test.sh
