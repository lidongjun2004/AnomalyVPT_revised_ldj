#!/bin/bash

# 定义固定的数据集和标志
dataset="mvtec"

# 定义配置名称
config_name="ViT-B-16"  

# 定义输出目录
output_dir="./output/train_vpt_${dataset}_${config_name}/"

# 执行训练命令
CUDA_VISIBLE_DEVICES=2 python main.py \
  --config-file "./configs/${config_name}.yaml" \
  --output-dir "$output_dir" \
  --name "$dataset" \
  --seed 1003 \
  --device 0 \
  --pixel

# --pixel