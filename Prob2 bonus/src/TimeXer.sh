#!/bin/bash

# source /opt/miniconda3/etc/profile.d/conda.sh
# conda activate tslib

export CUDA_VISIBLE_DEVICES=0
model_name=PatchTST
model_id=TimeXer_traffic
data_file=loop_sensor_train_aug.csv

# ========== Step 1: 训练模型 ==========
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --use_gpu True \
#   --gpu_type cuda \
#   --root_path ./dataset/ \
#   --data_path $data_file \
#   --model_id $model_id \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --target q \
#   --freq h \
#   --seq_len 24 \
#   --label_len 1 \
#   --pred_len 1 \
#   --e_layers 3 \
#   --factor 3 \
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 1 \
#   --d_model 128 \
#   --d_ff 256 \
#   --train_epochs 4 \
#   --batch_size 2048 \
#   --learning_rate 0.001 \
#   --des 'Exp' \
#   --itr 1 \
#   --use_amp \
#   --num_workers 12

# # ========== Step 2: 使用模型预测并保存 ==========
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --use_gpu True \
  --gpu_type cuda \
  --root_path ./dataset/ \
  --data_path loop_sensor_test_x_aug.csv \
  --seq_len 24 \
  --label_len 24 \
  --pred_len 1 \
  --model_id $model_id \
  --model $model_name \
  --data traffic \
  --features MS \
  --target q \
  --freq h \
  --e_layers 3 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 1 \
  --d_model 128 \
  --d_ff 256 \
  --batch_size 256 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1 \
  --num_workers 8 \
  --use_amp 
