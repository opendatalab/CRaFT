from mmengine.config import read_base

with read_base():
    from ..datasets.triviaqa.triviaqa_train_3shot_OE_kq import triviaqa_datasets as datasets

    from ..models.hf_llama.hf_llama3_8b_instruct import model_init as llama3_8b_instruct_model


    # from .models.m_triviaqa_en_train import models as m_triviaqa_en_train_model

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
generation_kwargs = dict(
    temperature=1.0,
    do_sample=True,
)
for m in models:
    m['generation_kwargs'] = generation_kwargs

work_dir = './results/Knowledge_Query/' + '_'.join(__file__.split('/')[-1].split('_')[:3]) + '/' # do NOT modify this line, yapf: disable, pylint: disable
