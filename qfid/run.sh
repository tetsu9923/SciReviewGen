#!/bin/bash

#$-l h_rt=96:00:00
#$-l rt_AG.small=1 
#$-o /groups/gcc50560/kasanishi/log
#$-j y
#$-cwd
#$-m a
#$-m b
#$-m e

source ~/.bashrc
source ~/.bash_profile
source /etc/profile.d/modules.sh
conda activate pegasus
export TRANSFORMERS_CACHE=/groups/gcc50560/kasanishi/huggingface_cache
module load fuse-sshfs/3.7.2 cuda/11.3/11.3.1 cudnn/8.4/8.4.1
sshfs hoth:/home/t-kasanishi/papersum /home/acc12378ha/hoth
cd /home/acc12378ha/hoth

# python ファイル名確認！

CUDA_VISIBLE_DEVICES=0 \
    python run_summarization_fid_normal.py \
    --model_name_or_path facebook/bart-large-cnn \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file /groups/gcc50560/kasanishi/csv/aquamuse_train_each.csv \
    --validation_file /groups/gcc50560/kasanishi/csv/aquamuse_val_each.csv \
    --test_file /groups/gcc50560/kasanishi/csv/aquamuse_test_each.csv \
    --text_column reference \
    --summary_column target \
    --output_dir /groups/gcc50560/kasanishi/fid_aquamuse_lr=5e-05 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --num_train_epochs=10 \
    --predict_with_generate \
    --save_strategy epoch \
    --learning_rate 5e-05 \
    --evaluation_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model=eval_rouge2 \
    --greater_is_better True \
    --save_total_limit=1 \
    --max_source_length 2048 \
    --max_target_length 512 \
    --generation_max_length 256 \
    --num_beams 4 \
    --seed 1 \
#    --pad_to_max_length
#    --overwrite_output_dir \
