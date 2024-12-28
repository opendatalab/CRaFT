from mmengine.config import read_base

with read_base():
    from ..datasets.nq.nq_dev_0shot_gen import nq_datasets as datasets

    from ..models.hf_llama.hf_llama3_8b_instruct import model_init as llama3_8b_instruct_model
    from ..models.hf_llama.hf_llama3_8b_instruct import model_triviaqa_CRaFT as llama3_8b_instruct_triviaqa_CRaFT_model

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

work_dir = './results/Eval/' + __file__.split('/')[-1].split('.')[0] + '/' # do NOT modify this line, yapf: disable, pylint: disable
