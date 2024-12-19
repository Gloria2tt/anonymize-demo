# 从节点 train_node1.sh
#!/bin/bash

export MASTER_ADDR="10.54.102.212"
export OUTPUT_DIR="./finetune_2block_finetuning_encodec8"
export MODEL_DIR="auffusion/auffusion-full-no-adapter"

# 添加日志记录
ACCELERATE_LOG_LEVEL=debug accelerate launch \
    --multi_gpu \
    --machine_rank=0 \
    --num_machines=2 \
    --num_processes=16 \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=29535 \
    --mixed_precision="fp16" \
    train_encodec.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --dataset_name=fusing/fill50k \
    --learning_rate=5e-5 \
    --max_train_steps=100000 \
    --train_batch_size=4 \
    --mixed_precision="fp16" \
    --tracker_project_name="controlnet-demo" \
    --report_to=tensorboard \
    --checkpointing_steps=10000 \
    --lr_warmup_steps=1000 \
    --adam_weight_decay=5e-2 \
    --gradient_accumulation_steps=1 \
    > node2_train.log 2>&1



