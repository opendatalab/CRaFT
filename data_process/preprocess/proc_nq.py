import os
import os.path as osp
import pandas as pd
from mmengine import load
from utils import df_to_list_of_dict, dump_to_json_file


def proc_nq():
    version = osp.splitext(osp.basename(__file__))[0]
    src_dir = 'dataset/download_dataset/natural-questions/nq_open'
    dst_dir = 'dataset/preprocessed_dataset/nq/en'
    os.makedirs(dst_dir, exist_ok=True)

    for split in ['dev', 'train']:
        src_file = osp.join(src_dir, f'NQ-open.{split}.jsonl')
        dst_csv = osp.join(dst_dir, f'{split}.csv')
        dst_json = osp.join(dst_dir, f'{split}.json')

        df = pd.read_json(src_file, lines=True)
        df['qid'] = pd.Series(df.index).apply(lambda x: f'nq_{split}_{x}')
        df.rename(
            columns={'answer': 'answers'},
            inplace=True,
        )
        df['answer'] = df['answers'].apply(lambda x: x[0])
        df['lang'] = 'en'
        df = df[['qid', 'lang', 'question', 'answer', 'answers']]

        df.to_csv(dst_csv, index=False)
        dump_to_json_file(df_to_list_of_dict(df), dst_json)


if __name__ == '__main__':
    proc_nq()
