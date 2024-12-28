import os
import os.path as osp
import pandas as pd
import random
from copy import deepcopy
import cv2
import numpy as np

from mmengine import load

from utils import df_to_list_of_dict, dump_to_json_file

random.seed(0)

class KqKfPostProcessor:

    def __init__(
        self,
        kq_init_dir,
        kq_rehearsal_dir, 
        dst_base_dir
    ):
        self.kq_init_dir = kq_init_dir
        self.kq_rehearsal_dir = kq_rehearsal_dir
        self.dst_base_dir = dst_base_dir
        os.makedirs(self.dst_base_dir, exist_ok=True)

        self._load_kf()

    def _load_kf(self):
        assert osp.isfile(self.kq_init_dir), f'{self.kq_init_dir} not exists.'
        kq_df = pd.DataFrame(load(self.kq_init_dir))
        kq_df = kq_df[[
            'qid', 'question', 'answer', 'correct', 'pred_emb_sim_mat_avg'
        ]]
        assert osp.isfile(self.kq_rehearsal_dir), f'{self.kq_rehearsal_dir} not exists.'
        rehearsal_kq_df = pd.DataFrame(load(self.kq_rehearsal_dir)['details']).T
        merged_df = pd.merge(kq_df, rehearsal_kq_df[['qid', 'is_correct']], on='qid', how='left')
        merged_df['correct_rehearsal'] = merged_df.groupby('qid')['is_correct'].transform('mean')
        kq_df['correct_rehearsal'] = merged_df['correct_rehearsal']
        self.kq_df = kq_df

    def _output_df(self, src_df, dst_dir, dst_prefix):
        assert osp.isdir(dst_dir)

        dst_file = osp.join(dst_dir, f'{dst_prefix}.json')
        assert not osp.isfile(dst_file), f'{dst_file} already exists, abort.'

        df = src_df[['qid', 'question', 'answer']]
        dump_to_json_file(df_to_list_of_dict(df), dst_file)

    def Van_Tuning(self, src_df, total_n):
        dst_dir = self.dst_base_dir

        df = deepcopy(src_df.sample(n=total_n, random_state=0))
        self._output_df(df, dst_dir, f'Van_Tuning_{total_n}')

    def Cor_RAIT(self, src_df, total_n):
        correct_thr1 = 0.49
        correct_thr2 = 0.51
        n_Idk = total_n // 10 * 2
        n_vanilla = total_n - n_Idk

        dst_dir = self.dst_base_dir
        dst_prefix = f'Cor_RAIT_Idk{n_Idk}_van{n_vanilla}_corThr_{correct_thr1}_{correct_thr2}'

        # ===>>> 根据correct选择样本
        vanilla_df = src_df[src_df['correct'] >= correct_thr2]
        Idk_df = src_df[src_df['correct'] <= correct_thr1]
        # <<<=== 根据correct选择样本

        assert vanilla_df.shape[0] >= n_vanilla
        assert Idk_df.shape[0] >= n_Idk

        # ===>>> 根据certainty选择样本
        vanilla_df = vanilla_df.sample(n=n_vanilla, random_state=0)
        Idk_df = Idk_df.sample(n=n_Idk, random_state=0)
        # <<<=== 根据certainty选择样本

        Idk_df['answer'] = 'I don\'t know'

        Cor_RAIT_df = pd.concat([vanilla_df, Idk_df])
        self._output_df(Cor_RAIT_df, dst_dir, dst_prefix)

    def Corcer_RAIT(self, src_df, total_n):  # TODO: rename to query1
        correct_thr1 = 0.49
        correct_thr2 = 0.51
        n_Idk = total_n // 10 * 2
        n_vanilla = total_n - n_Idk
        dst_dir = self.dst_base_dir
        dst_prefix = f'Corcer_RAIT_Idk{n_Idk}_van{n_vanilla}_corThr_{correct_thr1}_{correct_thr2}'

        # ===>>> 根据correct选择样本
        vanilla_df = src_df[src_df['correct'] >= correct_thr2]
        Idk_df = src_df[src_df['correct'] <= correct_thr1]
        # <<<=== 根据correct选择样本

        assert vanilla_df.shape[0] >= n_vanilla
        assert Idk_df.shape[0] >= n_Idk

        # ===>>> 根据certainty选择样本
        # vanilla_df, 选择similarity最大的
        vanilla_df.sort_values('pred_emb_sim_mat_avg', ascending=False, inplace=True)
        # Idk_df, 选择similarity最小的
        Idk_df.sort_values('pred_emb_sim_mat_avg', ascending=True, inplace=True)

        vanilla_df = vanilla_df.head(n_vanilla)
        Idk_df = Idk_df.head(n_Idk)
        # <<<=== 根据certainty选择样本

        Idk_df['answer'] = 'I don\'t know'

        Corcer_RAIT_df = pd.concat([vanilla_df, Idk_df])
        self._output_df(Corcer_RAIT_df, dst_dir, dst_prefix)


    def CRaFT(self, src_df, total_n):
        correct_thr1 = 0.49
        correct_thr2 = 0.51
        n_Idk = total_n // 10 * 2
        n_vanilla = total_n - n_Idk
        dst_dir = self.dst_base_dir
        dst_prefix = f'CRaFT_Idk{n_Idk}_van{n_vanilla}_corThr_{correct_thr1}_{correct_thr2}'

        # ===>>> 根据correct选择样本
        vanilla_df = src_df[src_df['correct'] >= correct_thr2]
        Idk_df = src_df[src_df['correct'] <= correct_thr1]
        # <<<=== 根据correct选择样本

        # ===>>> 根据dxdy选择样本
        Idk_df = Idk_df[Idk_df['correct_rehearsal'] - Idk_df['correct'] < 0]
        # <<<=== 根据dxdy选择样本

        assert vanilla_df.shape[0] >= n_vanilla
        assert Idk_df.shape[0] >= n_Idk

        # ===>>> 根据certainty选择样本
        # vanilla_df, 选择similarity最大的
        vanilla_df.sort_values('pred_emb_sim_mat_avg', ascending=False, inplace=True)
        # Idk_df, 选择similarity最小的
        Idk_df.sort_values('pred_emb_sim_mat_avg', ascending=True, inplace=True)

        vanilla_df = vanilla_df.head(n_vanilla)
        Idk_df = Idk_df.head(n_Idk)
        # <<<=== 根据certainty选择样本

        Idk_df['answer'] = 'I don\'t know'

        CRaFT_df = pd.concat([vanilla_df, Idk_df])
        self._output_df(CRaFT_df, dst_dir, dst_prefix)

    def process(self, total_n):
        self.Van_Tuning(self.kq_df, total_n)
        self.Cor_RAIT(self.kq_df, total_n)
        self.Corcer_RAIT(self.kq_df, total_n)
        self.CRaFT(self.kq_df, total_n)


def main():
    model = 'llama-3-8b-instruct-hf'
    results_base_dir = 'results/Knowledge_Query/KQ_triviaqa_train'
    latest_folder = max((os.path.join(results_base_dir, d) for d in os.listdir(results_base_dir) if os.path.isdir(os.path.join(results_base_dir, d))), key=os.path.getmtime)
    kq_init_dir = osp.join(latest_folder, f'kq_res/all-MiniLM-L6-v2/{model}/triviaqa_train-BASIC2_3shot_infer10/scores.json')
    kq_rehearsal_dir = osp.join(latest_folder, f'results/{model}-rehearsal-triviaqa/triviaqa_train-BASIC2_3shot_infer10.json')
    dst_base_dir = f'dataset/RAIT_dataset/triviaqa/{model}'
    processor = KqKfPostProcessor(kq_init_dir, kq_rehearsal_dir, dst_base_dir)
    processor.process(total_n=10000)

if __name__ == '__main__':
    main()
