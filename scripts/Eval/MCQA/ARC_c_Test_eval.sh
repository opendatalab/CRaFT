set -x

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 \
PYTHONPATH=. \
python compass_config/run.py \
    compass_config/Eval/eval_ARC_c_test.py \
    --dump-eval-details \
    --max-num-workers 128 \
    -r \

PYTHONPATH=. python compass_config/Summary/summarize.py ARC_Test
