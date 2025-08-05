#!/bin/bash
PYTHON="/home/fengwei/anaconda3/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

model='dino'

DATASETS=("cubc" "scarsc" "fgvcc")
TASKs=("A_L+A_U+B->A_U+B+C")

TASK=${TASKs[0]}
DATASET=${DATASETS[1]}

SAVE_DIR=/mnt/sdb/fengwei/GCD_natural/HiLo-main/checkpoints/
WEIGHTS_PATH=/mnt/sdb/fengwei/GCD_natural/HiLo-main/data/sample_weights/ssbc/scarsc.json

${PYTHON} -m methods.ours.mi_dis_pm \
            --dataset_name "scarsc" \
            --batch_size 150 \
            --grad_from_block 11 \
            --epochs 200 \
            --num_workers 8 \
            --sup_weight 0.35 \
            --weight_decay 5e-5 \
            --warmup_teacher_temp 0.07 \
            --teacher_temp 0.04 \
            --warmup_teacher_temp_epochs 30 \
            --memax_weight 2 \
            --transform 'imagenet' \
            --weights_path ${WEIGHTS_PATH} \
            --lr 0.05 \
            --eval_funcs 'v2' \
            --src_env 'N/A' \
            --tgt_env 'N/A' \
            --aux_env 'N/A' \
            --task_type ${TASK} \
            --model ${model} \
            --model_path ${SAVE_DIR}${DATASET}
