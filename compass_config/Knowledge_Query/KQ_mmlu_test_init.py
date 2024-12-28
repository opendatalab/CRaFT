from mmengine.config import read_base
from opencompass.models import HuggingFaceBaseModel

with read_base():
    from ..datasets.mmlu.mmlu_test_5shot_MC_kq import mmlu_datasets

    from ..models.hf_llama.hf_llama3_8b_instruct import model_init as llama3_8b_instruct_model

datasets = mmlu_datasets
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
for m in models:
    if str(m['type']) != 'HuggingFaceBaseModel':
        m['type'] = HuggingFaceBaseModel

work_dir = './results/Knowledge_Query/' + '_'.join(__file__.split('/')[-1].split('_')[:3]) + '/' # do NOT modify this line, yapf: disable, pylint: disable
