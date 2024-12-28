set -x

for CONFIG_FILE in \
llama3_8b_instruct_full_CRaFT_triviaqa_train_Idk2000_van8000__REFUSE1.py \
; do

# ---
CONFIG_NAME=$(basename $CONFIG_FILE .py)
WORK_DIR=ckpt/$CONFIG_NAME

# -------------------------------- convert to hf

XTUNER_PTH=$(cat ${WORK_DIR}/last_checkpoint)
PREFIX=${XTUNER_PTH%.pth}

CONFIG_NAME_OR_PATH=${WORK_DIR}/${CONFIG_FILE}
HF_PATH=${WORK_DIR}/last_ckpt_hf

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 \
xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${XTUNER_PTH} ${HF_PATH} &

sleep 1s

# -------------------------------- merge

NAME_OR_PATH_TO_ADAPTER=${HF_PATH}
MERGED_PATH=${WORK_DIR}/last_ckpt_hf_merged

if [[ $CONFIG_FILE == *"full"* ]]; # CONFIG_FILE contains "full"
then
    ln -snf ${NAME_OR_PATH_TO_ADAPTER} ${MERGED_PATH}
else
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 \
    xtuner convert merge \
        ${NAME_OR_PATH_TO_LLM} \
        ${NAME_OR_PATH_TO_ADAPTER} \
        ${MERGED_PATH} \
        --max-shard-size 20GB
fi

done