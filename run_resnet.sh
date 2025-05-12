#!/bin/bash

nohup python train_resnet_model.py \
  --dataset augmented \
  --pretrained 0 \
  --epoch 50 \
  --batch_size 256 \
  --learning_rate 0.01 \
  --resnet_model resnet18 \
  --dropout 0 \
  --optimizer SGD \
  --scheduler ReduceLROnPlateau \
  --early_stopping \
  --save_dir "resnet18_pretrained_0_bs256_lr001_dropout_0_SGD_ReduceLROnPlateau_earlystop_1" \
  > resnet18_pretrained_0_bs256_lr001_dropout_0_SGD_ReduceLROnPlateau_earlystop_1.log 2>&1 &


# 获取任务的进程ID（PID）
pid1=$!

# 等待任务完成
wait $pid1

# 输出信息，表明任务已经完成
echo "tasks are completed."
