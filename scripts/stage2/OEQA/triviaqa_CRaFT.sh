set -x

for CONFIG_FILE in \
train/triviaqa/llama3_8b_instruct_full_CRaFT_triviaqa_train_Idk2000_van8000__REFUSE1.py \
; do

CONFIG_NAME=$(basename $CONFIG_FILE .py)
WORK_DIR=ckpt/$CONFIG_NAME

# ----------------------------------------------------------
mkdir -p $WORK_DIR

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 \
PYTHONPATH=. \
NPROC_PER_NODE=4 xtuner train \
    $CONFIG_FILE \
    --deepspeed deepspeed_zero2 \
    --work-dir $WORK_DIR \

sleep 1s

done