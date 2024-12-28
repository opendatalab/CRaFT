import os
from copy import deepcopy
import pandas as pd

from utils import df_to_xlsx, cal_THS
from data_process.instructions_construction.mmlu_info import mmlu_groups

class MCQASummarizer:

    def __init__(
        self,
        *,
        dataset_name,
        latest_csv,
        dst_xlsx,
    ):
        self.dataset_name = dataset_name
        self.dataset_prefix = dataset_name + '-'
        self.latest_csv = latest_csv
        self.dst_xlsx = dst_xlsx

        self._load_latest_csv()
        self._auto_parse()
        

    def _load_latest_csv(self):
        src_df = pd.read_csv(self.latest_csv)
        src_df.fillna("", inplace=True)

        # --- drop rows not startswith dataset_prefix
        dataset_prefix = self.dataset_prefix
        src_df = src_df[src_df['dataset'].str.startswith(dataset_prefix)]

        # --- check duplicated dataset
        assert src_df[[
            'dataset', 'metric'
        ]].duplicated().sum() == 0, "duplicated dataset in sum files"

        drop_cols = ["version", "mode"]
        src_df = src_df.drop(columns=drop_cols)

        self.src_df = src_df
    

    def _auto_parse(self):
        datasets = self.src_df['dataset']
        self.src_df['task'] = datasets.apply(lambda x: x.split("-")[1])
        self.src_df['prompting'] = datasets.apply(lambda x: x.split("-")[2])

        self.src_df.drop(columns=['dataset'], inplace=True)
        self.src_df.set_index(['task', 'metric', 'prompting'],
                              inplace=True)
        self.src_df.columns.name = 'model'
        self.src_df = self.src_df.apply(pd.to_numeric, errors='raise')        
    

    def process(self):
        group_means = {}
        df = deepcopy(self.src_df)
        df = df.unstack(level=[1, 2])
        df = df.reorder_levels([0, 2, 1], axis=1)
        df.sort_index(axis=1, level=0, inplace=True)
        for group_name, group_tasks in mmlu_groups.items():
            group_df = df.loc[group_tasks]
            group_mean = group_df.mean(axis=0)
            group_means[group_name] = group_mean
        
        gm_df = pd.DataFrame(group_means).T
        gm_df.index.name = 'task'
        gm_df = gm_df.stack(level=[0, 1])

        init_tuple = tuple(gm_df.iloc[0][['correct_ratio', 'incorrect_ratio']])
        gm_df['THS'] = gm_df.apply(lambda row: cal_THS((row['correct_ratio'], row['incorrect_ratio']), init_tuple), axis=1)
        df_to_xlsx(gm_df,
                   self.dst_xlsx,
                   index=True,
                   text_wrap=True)

def main():
    dataset_name = "mmlu_val"
    resulr_base_dir = f"results/Eval/eval_{dataset_name}"
    latest_folder = max((os.path.join(resulr_base_dir, d, "summary") for d in os.listdir(resulr_base_dir) if os.path.isdir(os.path.join(resulr_base_dir, d))), key=os.path.getmtime)
    latest_csv = max(
        (os.path.join(latest_folder, f) for f in os.listdir(latest_folder) if f.endswith('.csv')),
        key=lambda x: os.path.getmtime(x)
    )
    dst_xlsx = latest_csv.replace('csv', 'xlsx')
    summarizer = MCQASummarizer(
        dataset_name=dataset_name,
        latest_csv=latest_csv,
        dst_xlsx=dst_xlsx,
    )
    summarizer.process()

if __name__ == '__main__':
    main()
