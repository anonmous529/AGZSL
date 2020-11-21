
lr=5e-4
opt_decay=5e-7
file='cublr'$lr'_opt'$opt_decay'_w20_s4_51Finetune'

CUDA_VISIBLE_DEVICES=1 python train_unseen.py --dataset CUB --ways 20 --shots 4 --lr $lr --opt_decay $opt_decay --step_size 500 --log_file $file --model_file $file'.pt'
