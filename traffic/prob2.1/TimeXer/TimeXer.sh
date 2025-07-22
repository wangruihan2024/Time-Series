#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
model_name=TimeXer

cd /Users/wang/Documents/StockForcast/Time-Series-Library-main

# ====== 1. 训练 ======
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path loop_sensor_train.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --target q \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --d_model 512 \
  --d_ff 512 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1 \
  --use_gpu True \
  --gpu_type mps

# ====== 2. 测试并输出到 output/1.csv ======
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/ \
  --data_path loop_sensor_test_x.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --target q \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --d_model 512 \
  --d_ff 512 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1 \
  --use_gpu True \
  --gpu_type mps