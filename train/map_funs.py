import json


def map_fn_mmlu(example):

    _task_name = example['prompt_task_name']
    _question = example['prompt_question']
    _answer = example['prompt_answer']
    _hint = example['prompt_instruction'].format(task_name=_task_name)

    question = example['question']
    A = example['A']
    B = example['B']
    C = example['C']
    D = example['D']
    target = example['target']

    return {
        'conversation': [{
            'system': '',
            'input':
            f'{_hint}\n{_question}: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n{_answer}: ',
            'output': f'{target}\n'
        }]
    }

def map_fn_triviaqa_BASIC1(example):
    question = example['question']
    answer = example['answer']

    return {
        'conversation': [{
            'system': '',
            'input': question,
            'output': answer
        }]
    }


def map_fn_triviaqa_BASIC2_3ShotFromTrain(example):
    question = example['question']
    answer = example['answer']

    prompt = "Answer the following question as simple as possible.\nQuestion: Who was President when the first Peanuts cartoon was published?\nAnswer:\nHarry Truman\n\nAnswer the following question as simple as possible.\nQuestion: Which American-born Sinclair won the Nobel Prize for Literature in 1930?\nAnswer:\nSinclair Lewis\n\nAnswer the following question as simple as possible.\nQuestion: Where in England was Dame Judi Dench born?\nAnswer:\nYork\n\nAnswer the following question as simple as possible.\nQuestion: {question}\nAnswer:"

    return {
        'conversation': [{
            'system': '',
            'input': prompt.format(question=question),
            'output': '\n' + answer
        }]
    }


def map_fn_triviaqa_REFUSE1(example):
    question = example['question']
    answer = example['answer']

    return {
        'conversation': [{
            'system': '',
            'input': "Answer the following question, and if you don't know the answer, only reply with 'I don't know': " + question,
            'output': answer
        }]
    } # yapf: disable
