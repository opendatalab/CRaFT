from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TriviaQADataset, TriviaQAEvaluator

with read_base():
    from .triviaqa_prompts import LANG_TO_INSTRUCTIONS, LANG_TO_ANSWER_PREFIX, LANG_TO_QUESTION_PREFIX

n_shot = '3shot'
n_infer = 10

train_split = 'dev'
test_split = 'train'
n_infer_dict = {
    train_split: 1,
    test_split: n_infer,
}

langs = ['en']
instruct_types = ['BASIC2']
# --------------------------------------------------------------
triviaqa_reader_cfg = dict(input_columns=['question'],
                           output_column='id_and_answers',
                           train_split=train_split,
                           test_split=test_split)
triviaqa_datasets = []
for instruct_type in instruct_types:
    prompting_name = instruct_type + '_' + n_shot + f'_infer{n_infer}'
    for lang in langs:
        _hint = LANG_TO_INSTRUCTIONS[instruct_type][lang]
        _question = LANG_TO_QUESTION_PREFIX[lang]
        _answer = LANG_TO_ANSWER_PREFIX[lang]
        triviaqa_infer_cfg = dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt=
                        f'{_hint}\n{_question}: {{question}}\n{_answer}:',
                    ),
                    dict(role='BOT', prompt='{answer}\n')
                ]),
            ),
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin='</E>',
                    round=[
                        dict(
                            role='HUMAN',
                            prompt=
                            f'{_hint}\n{_question}: {{question}}\n{_answer}:',
                        ),
                    ],
                ),
                ice_token='</E>',
            ),
            retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2]),
            inferencer=dict(type=GenInferencer, max_out_len=50),
        )

        triviaqa_eval_cfg = dict(
            evaluator=dict(
                type=TriviaQAEvaluator,
                splitters=[_hint[:40]],
            ),
            pred_role='BOT',
        )

        triviaqa_datasets.append(
            dict(
                type=TriviaQADataset,
                abbr=f'triviaqa_{test_split}-{prompting_name}',
                path='dataset/preprocessed_dataset/triviaqa',
                lang=lang,
                n_infer_dict=n_infer_dict,
                reader_cfg=triviaqa_reader_cfg,
                infer_cfg=triviaqa_infer_cfg,
                eval_cfg=triviaqa_eval_cfg))
