import os
import os.path as osp
import pandas as pd
import random
from copy import deepcopy
import cv2
import numpy as np
from glob import glob

from mmengine import load

from utils import df_to_list_of_dict, dump_to_json_file
from utils import draw_hist_mat

from mmlu_info import mmlu_task_names, LANG_TO_INSTRUCTIONS, LANG_TO_QUESTION_PREFIX, LANG_TO_ANSWER_PREFIX, mmlu_task_names

random.seed(42)

LANG_TO_IDK = {
    'en': "Sorry, I don't know."
}

class KqPostProcessor:

    def __init__(
        self,
        dataset_path, 
        results_dir,
        dst_dir,
        prompting
    ):
        self.correct_thr = 0.99
        self.total_n = 1000
        self.dst_file_path = osp.join(dst_dir, f'vanilla_cor{self.correct_thr}_n{self.total_n}.json')
        self.total_samples = []
        for task in mmlu_task_names:
            src_data_file = osp.join(dataset_path, f'{task}_test.csv') # raw_data
            src_result_file_cands = glob(
                osp.join(results_dir,
                         f'mmlu_test-{task}-*.json'))
            if len(src_result_file_cands) != 1:
                print(f'{src_result_file_cands=}')
                raise ValueError('len(src_result_file_cands) != 1')
            src_result_file = src_result_file_cands[0]
            
            data_df = pd.read_csv(src_data_file) # 导入数据
            data_df.fillna('None', inplace=True)
            results = load(src_result_file) # 导入预测结果
            assert 'details' in results, f'{results.keys()}'
            result_df = pd.DataFrame(results['details']).T

            data_df['prompt_task_name'] = mmlu_task_names[task]['en']
            data_df['prompt_instruction'] = LANG_TO_INSTRUCTIONS[
                prompting]['en']
            data_df['prompt_question'] = LANG_TO_QUESTION_PREFIX['en']
            data_df['prompt_answer'] = LANG_TO_ANSWER_PREFIX['en']
            for i, row in data_df.iterrows():
                result_df_row = result_df.iloc[i]
                entropy = result_df_row['origin_prediction']['entropy']
                corrrect = result_df_row['origin_prediction'][result_df_row['references']]['last_token_prob']
                if corrrect > self.correct_thr:
                    pass
                else:
                    data_df = data_df.drop(i)
            task_samples = data_df.to_dict(orient='records')
            self.total_samples.extend(task_samples)

    def process(self):
        dump_to_json_file(random.sample(self.total_samples, self.total_n), self.dst_file_path)


def main():
    model_name = 'llama-3-8b-instruct-hf'
    prompting = 'BASIC'

    results_base_dir = 'results/Knowledge_Query/KQ_mmlu_test'
    dataset_path = 'dataset/preprocessed_dataset/mmlu/test'
    latest_folder = max((os.path.join(results_base_dir, d) for d in os.listdir(results_base_dir) if os.path.isdir(os.path.join(results_base_dir, d))), key=os.path.getmtime)
    results_dir = osp.join(latest_folder, f'results/{model_name}')
    assert osp.isdir(results_dir), f'{results_dir} not exists'
    dst_dir = f"dataset/rehearsal_dataset/mmlu/{model_name}"
    os.makedirs(dst_dir, exist_ok=True)

    processor = KqPostProcessor(dataset_path, results_dir, dst_dir, prompting)
    processor.process()


if __name__ == '__main__':
    main()
