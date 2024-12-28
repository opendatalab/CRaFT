set -x

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 \
PYTHONPATH=. \
python compass_config/run.py \
    compass_config/Knowledge_Query/KQ_mmlu_test_init.py \
    --dump-eval-details \
    -r \
    --max-num-workers 128
