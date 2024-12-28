from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.mmlu import MMLUDataset, MMLU_HonestEvaluator

with read_base():
    from .mmlu_prompts import LANG_TO_INSTRUCTIONS, LANG_TO_ANSWER_PREFIX, LANG_TO_QUESTION_PREFIX, mmlu_task_names

ALL_LANGUAGES = ['en']

n_shot = '0shot'

train_split = 'test'
test_split = 'val'
mmlu_reader_cfg = dict(input_columns=['question', 'A', 'B', 'C', 'D'],
                         output_column='id_and_target',
                         train_split=train_split,
                         test_split=test_split)

mmlu_datasets = []
for instruct_type in ['BASIC', 'REFUSE']:
    prompting_name = instruct_type + '_' + n_shot
    for lang in ALL_LANGUAGES:
        _instructions = LANG_TO_INSTRUCTIONS[instruct_type]
        if lang not in _instructions:
            print(f'No instructions for {lang} in {instruct_type}')
            continue
        for _name in mmlu_task_names.keys():
            _task_name = mmlu_task_names[_name][lang]
            _hint = _instructions[lang].format(task_name=_task_name)
            _question = LANG_TO_QUESTION_PREFIX[lang]
            _answer = LANG_TO_ANSWER_PREFIX[lang]
            mmlu_infer_cfg = dict(
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(
                        begin='</E>',
                        round=[
                            dict(
                                role='HUMAN',
                                prompt=
                                f'{_hint}\n{_question}: {{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n{_answer}: '
                            ),
                        ],
                    ),
                    ice_token='</E>',
                ),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=GenInferencer),
            )

            mmlu_eval_cfg = dict(evaluator=dict(type=MMLU_HonestEvaluator))

            mmlu_datasets.append(
                dict(
                    abbr=f'mmlu_{test_split}-{_name}-{prompting_name}',
                    type=MMLUDataset,
                    path='dataset/preprocessed_dataset/mmlu',
                    name=_name,
                    reader_cfg=mmlu_reader_cfg,
                    infer_cfg=mmlu_infer_cfg,
                    eval_cfg=mmlu_eval_cfg,
                ))

        del _instructions, _name, _task_name, _hint, _question, _answer
