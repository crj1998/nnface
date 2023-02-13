#!/usr/bin/env bash
python -u train.py \
  --train_dataset /home/v-renjiechen/datasets/wider_yolo/WIDER_train \
  --val_dataset /home/v-renjiechen/datasets/wider_yolo/WIDER_val \
  --arch rfb \
  --num_epochs  130 \
  --milestones 60 100 \
  --lr  1e-2 \
  --batch_size  32 \
  --input_size  320 \
  --num_workers 8 \
