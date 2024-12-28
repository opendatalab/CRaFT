from opencompass.models import HuggingFacewithChatTemplate

model_init = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3-8b-instruct-hf',
        path='meta-llama/Meta-Llama-3-8B-Instruct',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
]

model_triviaqa_rehearsal = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3-8b-instruct-hf-rehearsal-triviaqa',
        path='ckpt/llama3_8b_instruct_full_rehearsal_triviaqa_train_cor0.995_cer0.995_n1000__BASIC1/last_ckpt_hf_merged',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
]

model_triviaqa_CRaFT = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3-8b-instruct-hf-CRaFT-triviaqa',
        path='ckpt/llama3_8b_instruct_full_CRaFT_triviaqa_train_Idk2000_van8000__REFUSE1/last_ckpt_hf_merged',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
]

model_mmlu_rehearsal = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3-8b-instruct-hf-rehearsal-mmlu',
        path='ckpt/llama3_8b_instruct_LoRA_rehearsal_mmlu_test_cor0.99_n1000__BASIC1/last_ckpt_hf_merged',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
]

model_mmlu_CRaFT = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3-8b-instruct-hf-CRaFT-mmlu',
        path='ckpt/llama3_8b_instruct_LoRA_CRaFT_mmlu_test_Idk1000_van4000__REFUSE1/last_ckpt_hf_merged',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
]
