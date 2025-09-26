#!/bin/bash

# 训练集
trainset="mvtec"

# 测试集
testset="visa"

# 配置名称
config_name="ViT-B-16-224"

# 输出目录
output_dir="./output/test_${testset}_${config_name}/"

# 模型文件路径模板
model_template="./output/train_vpt_${trainset}_${config_name}/model-%s.pth.tar"

model_names=()

# 初始值
start=3
end=30
step=3

# 循环生成数组元素
for ((i=start; i<=end; i+=step)); do
    model_names+=("ep$i")
done
model_names+=("latest")

# 循环遍历模型文件名数组
for model_name in "${model_names[@]}"; do
  # 生成完整的模型文件路径
  model=$(printf "$model_template" "$model_name")
  # echo "$model"
  # 执行测试命令
  CUDA_VISIBLE_DEVICES=2 python main.py \
    --config-file ./configs/${config_name}.yaml \
    --resume "$model" \
    --output-dir "${output_dir}/${model_name}/" \
    --name "$testset" \
    --eval \
    --device 0 \
    --vis \
    --pixel
done
