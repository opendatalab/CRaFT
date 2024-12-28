from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.datasets import MMLUDataset, MMLU_EditDistEvaluator
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator

with read_base():
    from .mmlu_prompts import LANG_TO_INSTRUCTIONS, LANG_TO_ANSWER_PREFIX, LANG_TO_QUESTION_PREFIX, mmlu_task_names

ALL_LANGUAGES = ['en']

n_shot = '5shot'

train_split = 'val'
test_split = 'test'
mmlu_reader_cfg = dict(input_columns=['question', 'A', 'B', 'C', 'D'],
                         output_column='target',
                         train_split=train_split,
                         test_split=test_split)

mmlu_datasets = []
for instruct_type in ['BASIC']:
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
            question_overall = '{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}'
            mmlu_infer_cfg = dict(
                ice_template=dict(
                    type=PromptTemplate,
                    template={opt: f'{_question}: {question_overall}\n{_answer}: {opt}\n' for opt in ['A', 'B', 'C', 'D']},
                ),
                prompt_template=dict(
                    type=PromptTemplate,
                    template={opt: f'{_hint}</E>{_question}: {question_overall}\n{_answer}: {opt}' for opt in ['A', 'B', 'C', 'D']},
                    ice_token='</E>',
                ),
                retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
                inferencer=dict(type=PPLInferencer),
            )

            mmlu_eval_cfg = dict(evaluator=dict(
                type=AccwithDetailsEvaluator, ))

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
