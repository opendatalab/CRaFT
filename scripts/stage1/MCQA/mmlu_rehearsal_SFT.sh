set -x

CONFIG_FILE=train/mmlu/llama3_8b_instruct_LoRA_rehearsal_mmlu_test_cor0.99_n1000__BASIC1.py

CONFIG_NAME=$(basename $CONFIG_FILE .py)
# 去掉用户扩展名

WORK_DIR=ckpt/$CONFIG_NAME

# ----------------------------------------------------------
mkdir -p $WORK_DIR
cp $0 $WORK_DIR

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 \
PYTHONPATH=. \
NPROC_PER_NODE=4 xtuner train \
    $CONFIG_FILE \
    --deepspeed deepspeed_zero2 \
    --work-dir $WORK_DIR \