set -x
CONFIG_FILE=llama3_8b_instruct_LoRA_CRaFT_mmlu_test_Idk1000_van4000__REFUSE1.py

NAME_OR_PATH_TO_LLM=meta-llama/Meta-Llama-3-8B-Instruct

# ---
CONFIG_NAME=$(basename $CONFIG_FILE .py)

WORK_DIR=ckpt/$CONFIG_NAME

# -------------------------------- convert to hf

XTUNER_PTH=$(cat ${WORK_DIR}/last_checkpoint)
PREFIX=${XTUNER_PTH%.pth}

CONFIG_NAME_OR_PATH=${WORK_DIR}/${CONFIG_FILE}
HF_PATH=${WORK_DIR}/last_ckpt_hf

xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${XTUNER_PTH} ${HF_PATH}

# -------------------------------- merge

NAME_OR_PATH_TO_ADAPTER=${HF_PATH}
MERGED_PATH=${WORK_DIR}/last_ckpt_hf_merged

xtuner convert merge \
    ${NAME_OR_PATH_TO_LLM} \
    ${NAME_OR_PATH_TO_ADAPTER} \
    ${MERGED_PATH} \
    --max-shard-size 20GB
