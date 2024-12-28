import os
import os.path as osp
import pandas as pd
import random
from copy import deepcopy
import cv2
import numpy as np

from mmengine import load

from utils import df_to_list_of_dict, dump_to_json_file
from utils import draw_hist_mat

random.seed(0)


class KqPostProcessor:

    def __init__(
        self,
        src_file,
        sim_name,
        dst_dir,
    ):
        self.src_file = src_file
        self.sim_name = sim_name
        self.dst_dir = dst_dir
        os.makedirs(dst_dir, exist_ok=True)

        df = pd.DataFrame(load(src_file))
        df = df[[
            'qid', 'question', 'answers', 'answer', 'pred', 'correct',
            'infer_id', 'is_correct', self.sim_name
        ]]
        self.df = df

    def _select_answer(self, ans_cands):
        if len(ans_cands) == 1:
            return ans_cands[0]

        # filter disambiguation
        wo_disambiguation = [
            ans for ans in ans_cands if 'disambiguation' not in ans
        ]
        if len(wo_disambiguation) == 0:
            return random.choice(ans_cands)
        elif len(wo_disambiguation) == 1:
            return wo_disambiguation[0]

        # filter all lower case
        wo_all_lower = [ans for ans in wo_disambiguation if not ans.islower()]
        if len(wo_all_lower) == 0:
            return random.choice(wo_disambiguation)
        elif len(wo_all_lower) == 1:
            return wo_all_lower[0]

        return random.choice(wo_all_lower)

    def _output_df(self, src_df, dst_prefix):
        dst_file = osp.join(self.dst_dir, f'{dst_prefix}.json')
        assert not osp.isfile(dst_file), f'{dst_file} already exists, abort.'

        df = src_df[['qid', 'question', 'answer']]
        dump_to_json_file(df_to_list_of_dict(df), dst_file)

    def _draw_hist_mat(self, src_df, dst_prefix):

        correct = src_df['correct']
        emb_sim = src_df[self.sim_name]
        dst_pngs = draw_hist_mat(
            correct,
            emb_sim,
            x_label='correct',
            y_label='emb_sim',
            dst_dir=self.dst_dir,
            dst_prefix=dst_prefix,
            hist_names=['num'],
            range=((0, 1), (0, 1)),
        )

        return dst_pngs[0]

    def replace_ans(self, row):
        kept_cols = ['is_correct', 'infer_id', 'pred']
        row_df = row[kept_cols].to_frame().T
        row_df = row_df[row_df['is_correct'] == 1]
        if row_df.shape[0] == 0:
            return row['answer']
        pred_vc = row_df['pred'].value_counts()
        return pred_vc.idxmax()

    def vanilla_correct_certain(self, src_df):
        correct_thr = 0.995
        similar_thr = 0.995
        total_n = 1000
        total_n_str = '_all' if total_n < 0 else str(total_n)
        dst_prefix = f'vanilla_cor{correct_thr}_cer{similar_thr}_n{total_n_str}'

        df = src_df[src_df['correct'] >= correct_thr].copy()
        df = df[df[self.sim_name] >= similar_thr]
        if total_n > 0:
            df = df.sample(n=total_n, random_state=0)

        df['answer'] = df.apply(self.replace_ans, axis=1)
        self._output_df(df, dst_prefix)
        # self._draw_hist_mat(df, dst_prefix)

    def process(self):

        self.vanilla_correct_certain(self.df)


def main():
    model_name = 'llama-3-8b-instruct-hf'

    embedding_model_name = 'all-MiniLM-L6-v2'
    sim_name = 'pred_emb_sim_mat_avg'
    # ----------------------------------------

    split = 'train'
    results_base_dir = 'results/Knowledge_Query/KQ_triviaqa_train'
    latest_folder = max((os.path.join(results_base_dir, d) for d in os.listdir(results_base_dir) if os.path.isdir(os.path.join(results_base_dir, d))), key=os.path.getmtime)
    src_root_dir = osp.join(latest_folder, f'kq_res/{embedding_model_name}/{model_name}/triviaqa_{split}-BASIC2_3shot_infer10')
    
    # ----------------------------------------

    src_file = osp.join(src_root_dir, 'scores.json')
    dst_dir = f"dataset/rehearsal_dataset/triviaqa/{model_name}"
    os.makedirs(dst_dir, exist_ok=True)

    processor = KqPostProcessor(src_file, sim_name, dst_dir)
    processor.process()


if __name__ == '__main__':
    main()
