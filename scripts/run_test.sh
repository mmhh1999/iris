#!/bin/bash

# data folder
DATASET_ROOT='/home/ubuntu/datasets/data'
DATASET='scannetpp'
# scene name
SCENE='45b0dac5e3'
LDR_IMG_DIR='Image'
EXP='scannetpp_bathroom2'
VAL_FRAME=0
CRF_BASIS=3
RES_SCALE=0.5
# whether has part segmentation
HAS_PART=0
SPP=64  # 降低采样数以加快测试速度
spp=16

# bake surface light field (SLF)
python slf_bake.py --dataset_root $DATASET_ROOT --scene $SCENE\
        --output checkpoints/$EXP/bake --res_scale $RES_SCALE\
        --dataset $DATASET

# extract emitter mask
python extract_emitter_ldr.py \
        --dataset_root $DATASET_ROOT --scene $SCENE\
        --output checkpoints/$EXP/bake --dataset $DATASET --res_scale $RES_SCALE\
        --threshold 0.99 

# initialize
python initialize.py --experiment_name $EXP --max_epochs 5 \
        --dataset $DATASET $DATASET_ROOT --scene $SCENE \
        --voxel_path checkpoints/$EXP/bake/vslf.npz \
        --emitter_path checkpoints/$EXP/bake/emitter.pth \
        --has_part $HAS_PART --val_frame $VAL_FRAME\
        --SPP $SPP --spp $spp --crf_basis $CRF_BASIS --res_scale $RES_SCALE