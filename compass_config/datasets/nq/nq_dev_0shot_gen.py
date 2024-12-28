from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TriviaQADataset, TriviaQAEvaluator

with read_base():
    from ..triviaqa.triviaqa_prompts import LANG_TO_INSTRUCTIONS, LANG_TO_ANSWER_PREFIX, LANG_TO_QUESTION_PREFIX

split = 'dev'
langs = ['en']
instruct_types = ['BASIC1', 'REFUSE1']
# --------------------------------------------------------------
triviaqa_reader_cfg = dict(input_columns=['question'],
                           output_column='id_and_answers',
                           train_split=split,
                           test_split=split)
nq_datasets = []
for instruct_type in instruct_types:
    prompting_name = instruct_type
    for lang in langs:
        triviaqa_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt=
                        f'{LANG_TO_INSTRUCTIONS[instruct_type][lang]}{{question}}?',
                    )
                ], )),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=50),
        )

        triviaqa_eval_cfg = dict(evaluator=dict(type=TriviaQAEvaluator),
                                 pred_role='BOT')

        nq_datasets.append(
            dict(
                type=TriviaQADataset,
                abbr=f'nq_{split}-{prompting_name}',
                path=
                'dataset/preprocessed_dataset/nq',
                lang=lang,
                reader_cfg=triviaqa_reader_cfg,
                infer_cfg=triviaqa_infer_cfg,
                eval_cfg=triviaqa_eval_cfg))
