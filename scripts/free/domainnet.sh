#!/bin/bash
PYTHON="../anaconda3/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

model='dino'

DATASETS=("domainnet")

ENVs1=("real")
# ENVs2=("clipart")
# ENVs2=("clipart" "infograph" "painting" "quickdraw" "sketch")
# ENVs2=("infograph")
ENVs2=("painting")
# ENVs2=("quickdraw")
# ENVs2=("sketch")

TASKs=("A_L+A_U+B->A_U+B+C")

partial_dataset='False'
uda_loss='True'

SAVE_DIR=../GCD_natural/free-main/checkpoints/free/

for d in ${!DATASETS[@]}; do
    for t in ${!TASKs[@]}; do
        for e in ${!ENVs1[@]}; do
            for ee in ${!ENVs2[@]}; do
                if [ ${ENVs1[$e]} != ${ENVs2[$ee]} ]; then
                    echo ${ENVs1[$e]} ${ENVs2[$ee]} others
                    WEIGHTS_PATH=../GCD_natural/free-main/data/sample_weights/domainnet/${ENVs2[$ee]}.json
                    ${PYTHON} -m methods.free.train_nips \
                                --dataset_name 'domainnet' \
                                --batch_size 800 \
                                --grad_from_block 11 \
                                --epochs 200 \
                                --num_workers 8 \
                                --sup_weight 0.35 \
                                --weight_decay 5e-5 \
                                --warmup_teacher_temp 0.07 \
                                --teacher_temp 0.04 \
                                --warmup_teacher_temp_epochs 30 \
                                --memax_weight 2 \
                                --transform 'domainnet' \
                                --weights_path ${WEIGHTS_PATH} \
                                --lr 0.05 \
                                --eval_funcs 'v2' \
                                --src_env ${ENVs1[$e]} \
                                --aux_env ${ENVs2[$ee]} \
                                --task_type ${TASKs[$t]} \
                                --use_partial_dataset ${partial_dataset} \
                                --use_uda_loss ${uda_loss} \
                                --model ${model} \
                                --model_path ${SAVE_DIR}${ENVs1[$e]}-${ENVs2[$ee]}-others
                else
                    continue
                fi
            done
        done
    done
done
