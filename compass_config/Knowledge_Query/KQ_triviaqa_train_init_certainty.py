import os, sys
import os.path as osp
from copy import deepcopy
import pandas as pd
import argparse
import torch
from tqdm import tqdm
import glob
from mmengine import load, dump

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import df_to_list_of_dict, dump_to_json_file
from utils import draw_hist_mat
from utils import list_of_list_to_list_of_str, list_of_str_to_list_of_list
from utils import MySentenceTransformer as SE

PRED_BLACK_LIST = ['<REFUSE>', '']

N_INFER = 10

tqdm.pandas()

class KnowledgeQueryOE:
    def __init__(
        self,
        *,
        dataset_name,
        results_dir,
        dst_root_dir,
        models=None,
        promptings=None,
        embedding_model_name='all-MiniLM-L6-v2',
    ):
        self.dataset_name = dataset_name
        self.results_dir = results_dir
        self.models = models
        self.promptings = promptings
        self.dst_root_dir = dst_root_dir
        self.embedding_model_name = embedding_model_name

        self.dst_root_dir = osp.join(dst_root_dir, embedding_model_name)
        os.makedirs(self.dst_root_dir, exist_ok=True)

        self._load_dataset()
        self._load_results()

    def _load_dataset(self):

        print(f'*' * 40)
        print(f'Loading dataset {self.dataset_name}...')
        print(f'*' * 40)
        
        if self.dataset_name == 'triviaqa_dev':
            dataset_df = pd.read_csv(
                'dataset/preprocessed_dataset/triviaqa/en/dev.csv'
            )
        elif self.dataset_name == 'triviaqa_train':
            dataset_df = pd.read_csv(
                'dataset/preprocessed_dataset/triviaqa/en/train.csv'
            )
        elif self.dataset_name == 'nq_dev':
            dataset_df = pd.read_csv(
                '/mnt/hwfile/opendatalab/wj/LLM/opendata/nq/nq-proc_v1/en/dev.csv'
            )
        else:
            raise NotImplementedError

        self.dataset_df = dataset_df

    def _load_results(self):
        if self.models is not None:
            assert set(self.models).issubset(os.listdir(self.results_dir))
        else:
            self.models = os.listdir(self.results_dir)

        print(f'Models: {self.models}')

        # ---------------------------------------- load
        dfs = {}
        promptings = []
        for model in self.models:
            model_dir = osp.join(self.results_dir, model)
            assert osp.isdir(model_dir), f"{model_dir} does not exist"

            res_files = [
                _ for _ in os.listdir(model_dir) if _.endswith('.json')
            ]

            res_files = [
                _ for _ in res_files if _.startswith(self.dataset_name + '-')
            ]
            assert len(res_files) > 0, f"{model_dir} has no result files"

            for res_file in res_files:
                filename = osp.splitext(res_file)[0]
                filename_split = filename.split('-')
                assert len(filename_split) == 2
                dataset_name = filename_split[0]
                prompting = filename.split('-')[-1]

                data = load(osp.join(model_dir, res_file))
                assert 'details' in data

                df = pd.DataFrame(data['details']).T
                dfs[(model, prompting)] = df
                promptings.append(prompting)

        self.promptings = list(set(promptings))

        self.result_dfs = dfs

    def preprocess(self, dataset_df, result_df, dst_dir):
        kept_result_cols = ['split_pred', 'is_correct', 'qid', 'infer_id']
        result_df = result_df[kept_result_cols]
        result_df['pred'] = result_df['split_pred'].apply(lambda x: x.strip())
        result_df = result_df.groupby('qid').agg(list)

        # check infer_id
        expected_infer_ids = [str(_) for _ in list(range(N_INFER))]
        check_infer_id = result_df['infer_id'].apply(
            lambda x: set(x) == set(expected_infer_ids))
        assert check_infer_id.all()

        assert dataset_df.shape[0] == result_df.shape[0]
        df = pd.merge(dataset_df, result_df, left_on='qid', right_index=True)

        self.gen_cols = ['pred']

        dst_prefix = osp.join(dst_dir, 'preproc')
        df.to_csv(f'{dst_prefix}.csv', index=False)
        dump_to_json_file(df_to_list_of_dict(df), f'{dst_prefix}.json')

        return df

    def get_embedding(self, df):
        for model in self.models:
            self.results_dir
            sentence_embedding_dir = os.path.join("/".join(self.results_dir.split('/')[:-1]), "sentence_embedding", model)
            with SE(self.embedding_model_name, sentence_embedding_dir) as se:
                for col in self.gen_cols:
                    df[col + '_emb'] = df[col].progress_apply(lambda x: se.encodes(x))
                    df[col + '_emb_sim_mat'] = df[col + '_emb'].apply(
                        lambda x: se.similarity(x, x))
        return df

    def cal_scores(self, df, dst_dir):
        dst_prefix = osp.join(dst_dir, 'scores')
        dst_csv = f'{dst_prefix}.csv'
        dst_json = f'{dst_prefix}.json'

        if osp.isfile(dst_csv) and osp.isfile(dst_json):
            print(f'Reuse {dst_json}.')
            df = pd.DataFrame(load(dst_json))
            for col in df.columns:
                if col.endswith('_emb_sim_mat'):
                    df[col] = df[col].apply(
                        lambda x: list_of_str_to_list_of_list(x))

            return df

        print(f'{pd.Timestamp.now()}: start calculating scores...')
        df = self.get_embedding(df)
        print(f'{pd.Timestamp.now()}: calculating scores finished.')

        df['correct'] = df['is_correct'].apply(lambda x: sum(x) / len(x))

        df['pred_lab_counts'] = df['is_correct'].apply(
            lambda x: {k: x.count(k)
                       for k in set(x)})
        df['pred_lab_entropy'] = df['pred_lab_counts'].apply(lambda x: -sum([
            v / sum(x.values()) * np.log(v / sum(x.values()))
            for v in x.values()
        ]))

        def drop_diag_then_avg(x):
            if isinstance(x, torch.Tensor):
                x = x.numpy()
            assert x.shape[0] == x.shape[1]

            diag_mask = np.eye(x.shape[0], dtype=bool)
            x_wo_diag = x.copy()
            x_wo_diag[diag_mask] = np.nan

            return np.nanmean(x_wo_diag)

        for col in self.gen_cols:
            src_col = col + '_emb_sim_mat'
            dst_col = col + '_emb_sim_mat_avg'
            df[dst_col] = df[src_col].apply(drop_diag_then_avg)

        cols = [_ for _ in df.columns if not _.endswith('_emb')]
        vis_df = df[cols]
        for col in vis_df.columns:
            if col.endswith('_emb_sim_mat'):
                vis_df[col] = vis_df[col].apply(lambda x: x.tolist())
                vis_df[col] = vis_df[col].apply(
                    lambda x: list_of_list_to_list_of_str(x))
            if col.endswith('_emb_sim_mat_avg'):
                vis_df[col] = vis_df[col].apply(lambda x: round(x, 2))

        vis_df.to_csv(dst_csv, index=False)
        dump_to_json_file(df_to_list_of_dict(vis_df), dst_json)

        return df

    def _vis_consist(self, df, x_col, y_col, dst_dir):
        x = np.array(df[x_col])
        y = np.array(df[y_col])

        draw_hist_mat(
            x,
            y,
            x_label=x_col,
            y_label=y_col,
            dst_dir=dst_dir,
            dst_prefix=f'{x_col}_vs_{y_col}_',
            x_bins=7,
            y_bins=5,
            figsize=(6, 5),
            range=((0, 1), (0, 1)),
            title=f'{dst_dir[-80:]}',
        )

    def vis_consist(self, df, dst_dir):
        self._vis_consist(df, 'correct', 'pred_lab_entropy', dst_dir)
        for col in self.gen_cols:
            self._vis_consist(df, 'correct', col + '_emb_sim_mat_avg', dst_dir)

    def process(self, n=-1):
        for model in self.models:
            for prompting in self.promptings:
                print('-' * 30 +
                      f' Processing {model}-{prompting}...')
                dst_dir = osp.join(
                    self.dst_root_dir, model,
                    f'{self.dataset_name}-{prompting}')
                os.makedirs(dst_dir, exist_ok=True)
                if n > 0:
                    dataset_df = deepcopy(self.dataset_df.head(n))
                else:
                    dataset_df = deepcopy(self.dataset_df)
                result_df = self.result_dfs[(model, prompting)]
                df = self.preprocess(dataset_df, result_df, dst_dir)
                df = self.cal_scores(df, dst_dir)
                self.vis_consist(df, dst_dir)


def main():
    promptings = None
    models = None

    # -----------------------------------------------------
    models = ['llama-3-8b-instruct-hf']
    dataset_name = 'triviaqa_train'
    results_base_dir = 'results/Knowledge_Query/KQ_triviaqa_train'
    latest_folder = max((os.path.join(results_base_dir, d) for d in os.listdir(results_base_dir) if os.path.isdir(os.path.join(results_base_dir, d))), key=os.path.getmtime)
    results_dir = osp.join(latest_folder, 'results')

    # -------------------------------------
    assert osp.isdir(results_dir), f"{results_dir} does not exist"
    assert models is None or isinstance(models, list)

    dst_root_dir = osp.join(osp.dirname(results_dir), 'kq_res')

    ic = KnowledgeQueryOE(
        dataset_name=dataset_name,
        results_dir=results_dir,
        models=models,
        promptings=promptings,
        dst_root_dir=dst_root_dir,
    )
    ic.process()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 21675))
        print('Waiting for debugger attach')
        debugpy.wait_for_client()
        debugpy.breakpoint()
        print('break on this line')

    main()
