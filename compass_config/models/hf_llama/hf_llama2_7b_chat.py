from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-2-7b-chat-hf-bf16',
        path='meta-llama/Llama-2-7b-chat-hf',
        model_kwargs=dict(torch_dtype='torch.bfloat16'),
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
