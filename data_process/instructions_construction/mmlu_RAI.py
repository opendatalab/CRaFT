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

random_seed = 42

LANG_TO_IDK = {
    'en': "I don't know."
}

class KqKfPostProcessor:

    def __init__(
        self,
        dataset_path, 
        results_init_dir,
        results_rehearsal_dir,
        dst_dir,
        prompting
    ):
        self.dst_dir = dst_dir
        self.all_data_df = pd.DataFrame()
        self.all_result_init_df = pd.DataFrame()
        self.all_result_rehearsal_df = pd.DataFrame()
        for task in mmlu_task_names:
            src_data_file = osp.join(dataset_path, f'{task}_test.csv') # raw_data
            data_df = pd.read_csv(src_data_file) # 导入数据
            data_df.fillna('None', inplace=True)
            data_df['prompt_task_name'] = mmlu_task_names[task]['en']
            data_df['prompt_instruction'] = LANG_TO_INSTRUCTIONS[
                prompting]['en']
            data_df['prompt_question'] = LANG_TO_QUESTION_PREFIX['en']
            data_df['prompt_answer'] = LANG_TO_ANSWER_PREFIX['en']
            self.all_data_df = pd.concat([self.all_data_df, data_df], ignore_index=True)

            src_result_init_file_cands = glob(
                osp.join(results_init_dir,
                         f'mmlu_test-{task}-*.json'))
            if len(src_result_init_file_cands) != 1:
                print(f'{src_result_init_file_cands=}')
                raise ValueError('len(src_result_init_file_cands) != 1')
            src_result_init_file = src_result_init_file_cands[0]
            
            results_init = load(src_result_init_file) # 导入预测结果
            assert 'details' in results_init, f'{results_init.keys()}'
            result_init_df = pd.DataFrame(results_init['details']).T
            result_init_df['id'] = task + '/test/' + result_init_df.index.astype(str)
            self.all_result_init_df = pd.concat([self.all_result_init_df, result_init_df], ignore_index=True)
            
            src_result_rehearsal_file_cands = glob(
                osp.join(results_rehearsal_dir,
                         f'mmlu_test-{task}-*.json'))
            if len(src_result_rehearsal_file_cands) != 1:
                print(f'{src_result_rehearsal_file_cands=}')
                raise ValueError('len(src_result_rehearsal_file_cands) != 1')
            src_result_rehearsal_file = src_result_rehearsal_file_cands[0]
            results_rehearsal = load(src_result_rehearsal_file) # 导入预测结果
            assert 'details' in results_rehearsal, f'{results_rehearsal.keys()}'
            result_rehearsal_df = pd.DataFrame(results_rehearsal['details']).T
            result_rehearsal_df['id'] = task + '/test/' + result_rehearsal_df.index.astype(str)
            self.all_result_rehearsal_df = pd.concat([self.all_result_rehearsal_df, result_rehearsal_df], ignore_index=True)
            
        self.all_result_init_df['entropy'] = self.all_result_init_df['origin_prediction'].apply(lambda row:row['entropy'])
        # norm ABCD token
        for col in ['A','B','C','D']:
            self.all_result_init_df[col] = self.all_result_init_df['origin_prediction'].apply(lambda row:row[col]['last_token_prob'])
        
        for col in ['A','B','C','D']:
            self.all_result_init_df[col] = self.all_result_init_df[col]/self.all_result_init_df.apply(lambda row:row['A']+row['B']+row['C']+row['D'],axis=1)
        
        self.all_result_init_df['correct'] = self.all_result_init_df.apply(lambda row:row[row['references']],axis=1)
        self.all_result_rehearsal_df['entropy'] = self.all_result_rehearsal_df['origin_prediction'].apply(lambda row:row['entropy'])
        # norm ABCD token
        for col in ['A','B','C','D']:
            self.all_result_rehearsal_df[col] = self.all_result_rehearsal_df['origin_prediction'].apply(lambda row:row[col]['last_token_prob'])
        
        for col in ['A','B','C','D']:
            self.all_result_rehearsal_df[col] = self.all_result_rehearsal_df[col]/self.all_result_rehearsal_df.apply(lambda row:row['A']+row['B']+row['C']+row['D'],axis=1)
        
        self.all_result_rehearsal_df['correct'] = self.all_result_rehearsal_df.apply(lambda row:row[row['references']],axis=1)

    def _output_df(self, src_df, dst_prefix):
        assert osp.isdir(self.dst_dir)

        dst_file = osp.join(self.dst_dir, f'{dst_prefix}.json')
        assert not osp.isfile(dst_file), f'{dst_file} already exists, abort.'

        dump_to_json_file(df_to_list_of_dict(src_df), dst_file)

    def Van_Tuning(self, n_van):
        Van_Tuning_df = deepcopy(self.all_data_df)
        sample_data_df = Van_Tuning_df.sample(n=n_van, replace=True, random_state=random_seed)
        self._output_df(sample_data_df, f'Van_Tuning_van{n_van}')

    def Cor_RAIT(self, n_idk, n_van):
        Cor_RAIT_df = deepcopy(self.all_data_df)
        Cor_RAIT_df['KQ'] = None
        for i, row in Cor_RAIT_df.iterrows():
            qid = row['id']
            qid_init_results = self.all_result_init_df[self.all_result_init_df['id'] == qid]
            qid_rehearsal_results = self.all_result_rehearsal_df[self.all_result_rehearsal_df['id'] == qid] 
            if qid_init_results.empty or qid_rehearsal_results.empty:  # 有些数据在result infer中被丢掉了
                continue
            qid_init_result = qid_init_results.iloc[0]    
            qid_rehearsal_result = qid_rehearsal_results.iloc[0]
            Cor_RAIT_df.at[i, 'KQ'] = {
                        'correct_init': qid_init_result.correct,
                        'entropy_init': qid_init_result.entropy,
                        'correct_rehearsal': qid_rehearsal_result.correct,
                        'entropy_rehearsal': qid_rehearsal_result.entropy
                    }
            if qid_init_result.correct >= 0.5:
                Cor_RAIT_df.loc[i, 'strategy'] = 'learn'
            else:
                Cor_RAIT_df.loc[i, 'target'] = LANG_TO_IDK['en']
                Cor_RAIT_df.loc[i, 'strategy'] = 'idk'
        
        learn_data_df = Cor_RAIT_df[Cor_RAIT_df['strategy'] == 'learn']
        idk_data_df = Cor_RAIT_df[Cor_RAIT_df['strategy'] == 'idk']

        sample_learn_data_df = learn_data_df.sample(n=n_van, replace=True, random_state=random_seed)
        sample_idk_data_df = idk_data_df.sample(n=n_idk, replace=True, random_state=random_seed)

        # 最终合并并打乱数据
        sample_data_df = pd.concat([sample_learn_data_df, sample_idk_data_df], ignore_index=True).sample(frac=1) # shuffle
        self._output_df(sample_data_df, f'Cor_RAIT_idk{n_idk}_van{n_van}')

    def CorCer_RAIT(self, n_idk, n_van):
        CorCer_RAIT_df = deepcopy(self.all_data_df)
        CorCer_RAIT_df['KQ'] = None
        for i, row in CorCer_RAIT_df.iterrows():
            qid = row['id']
            qid_init_results = self.all_result_init_df[self.all_result_init_df['id'] == qid]
            qid_rehearsal_results = self.all_result_rehearsal_df[self.all_result_rehearsal_df['id'] == qid] 
            if qid_init_results.empty or qid_rehearsal_results.empty:  # 有些数据在result infer中被丢掉了
                continue
            qid_init_result = qid_init_results.iloc[0]    
            qid_rehearsal_result = qid_rehearsal_results.iloc[0]
            CorCer_RAIT_df.at[i, 'KQ'] = {
                        'correct_init': qid_init_result.correct,
                        'entropy_init': qid_init_result.entropy,
                        'correct_rehearsal': qid_rehearsal_result.correct,
                        'entropy_rehearsal': qid_rehearsal_result.entropy
                    }
            if qid_init_result.correct >= 0.5:
                CorCer_RAIT_df.loc[i, 'strategy'] = 'learn'
            else:
                CorCer_RAIT_df.loc[i, 'target'] = LANG_TO_IDK['en']
                CorCer_RAIT_df.loc[i, 'strategy'] = 'idk'
        
        learn_data_df = CorCer_RAIT_df[CorCer_RAIT_df['strategy'] == 'learn']
        idk_data_df = CorCer_RAIT_df[CorCer_RAIT_df['strategy'] == 'idk']
        # 提取 entropy_init 列
        learn_data_df['entropy_init'] = learn_data_df['KQ'].apply(lambda x: x['entropy_init'])
        idk_data_df['entropy_init'] = idk_data_df['KQ'].apply(lambda x: x['entropy_init'])
        # stratified sampling 5000
        # 按照 ["info_vectors"]["y"] 最小值选择前4000个样本
        sample_learn_data_df = learn_data_df.nsmallest(n_van, "entropy_init", keep='all')

        # 按照 ["info_vectors"]["y"] 最大值选择后1000个样本
        sample_idk_data_df = idk_data_df.nlargest(n_idk, "entropy_init", keep='all')

        # 如果数量不足，则进行重复采样以补足数量
        if sample_learn_data_df.shape[0] < n_van:
            sample_learn_data_df = sample_learn_data_df.sample(n=n_van, replace=True, random_state=random_seed)
        if sample_idk_data_df.shape[0] < n_idk:
            sample_idk_data_df = sample_idk_data_df.sample(n=n_idk, replace=True, random_state=random_seed)

        # 最终合并并打乱数据
        sample_data_df = pd.concat([sample_learn_data_df, sample_idk_data_df], ignore_index=True).sample(frac=1) # shuffle
        self._output_df(sample_data_df, f'CorCer_RAIT_idk{n_idk}_van{n_van}')

    def CRaFT(self, n_idk, n_van):
        CRaFT_df = deepcopy(self.all_data_df)
        CRaFT_df['KQ'] = None
        for i, row in CRaFT_df.iterrows():
            qid = row['id']
            qid_init_results = self.all_result_init_df[self.all_result_init_df['id'] == qid]
            qid_rehearsal_results = self.all_result_rehearsal_df[self.all_result_rehearsal_df['id'] == qid] 
            if qid_init_results.empty or qid_rehearsal_results.empty:  # 有些数据在result infer中被丢掉了
                continue
            qid_init_result = qid_init_results.iloc[0]    
            qid_rehearsal_result = qid_rehearsal_results.iloc[0]
            CRaFT_df.at[i, 'KQ'] = {
                        'correct_init': qid_init_result.correct,
                        'entropy_init': qid_init_result.entropy,
                        'correct_rehearsal': qid_rehearsal_result.correct,
                        'entropy_rehearsal': qid_rehearsal_result.entropy
                    }
            if qid_init_result.correct >= 0.5:
                CRaFT_df.loc[i, 'strategy'] = 'learn'
            elif qid_rehearsal_result.correct - qid_init_result.correct > 0:
                CRaFT_df = CRaFT_df.drop(i)
            else:
                CRaFT_df.loc[i, 'target'] = LANG_TO_IDK['en']
                CRaFT_df.loc[i, 'strategy'] = 'idk'
        
        learn_data_df = CRaFT_df[CRaFT_df['strategy'] == 'learn']
        idk_data_df = CRaFT_df[CRaFT_df['strategy'] == 'idk']
        # 提取 entropy_init 列
        learn_data_df['entropy_init'] = learn_data_df['KQ'].apply(lambda x: x['entropy_init'])
        idk_data_df['entropy_init'] = idk_data_df['KQ'].apply(lambda x: x['entropy_init'])
        # stratified sampling 5000
        # 按照 ["info_vectors"]["y"] 最小值选择前4000个样本
        sample_learn_data_df = learn_data_df.nsmallest(n_van, "entropy_init", keep='all')

        # 按照 ["info_vectors"]["y"] 最大值选择后1000个样本
        sample_idk_data_df = idk_data_df.nlargest(n_idk, "entropy_init", keep='all')

        # 如果数量不足，则进行重复采样以补足数量
        if sample_learn_data_df.shape[0] < n_van:
            sample_learn_data_df = sample_learn_data_df.sample(n=n_van, replace=True, random_state=random_seed)
        if sample_idk_data_df.shape[0] < n_idk:
            sample_idk_data_df = sample_idk_data_df.sample(n=n_idk, replace=True, random_state=random_seed)

        # 最终合并并打乱数据
        sample_data_df = pd.concat([sample_learn_data_df, sample_idk_data_df], ignore_index=True).sample(frac=1) # shuffle
        self._output_df(sample_data_df, f'CRaFT_idk{n_idk}_van{n_van}')
        

    def process(self):
        n_idk, n_van = 1000, 4000
        # self.Van_Tuning(n_idk + n_van)
        self.Cor_RAIT(n_idk, n_van)
        # self.CorCer_RAIT(n_idk, n_van)
        # self.CRaFT(n_idk, n_van)


def main():
    model_name = 'llama-3-8b-instruct-hf'
    prompting = 'REFUSE'

    results_base_dir = 'results/Knowledge_Query/KQ_mmlu_test'
    dataset_path = 'dataset/preprocessed_dataset/mmlu/test'
    latest_folder = max((os.path.join(results_base_dir, d) for d in os.listdir(results_base_dir) if os.path.isdir(os.path.join(results_base_dir, d))), key=os.path.getmtime)
    results_init_dir = osp.join(latest_folder, f'results/{model_name}')
    assert osp.isdir(results_init_dir), f'{results_init_dir} not exists'
    results_rehearsal_dir = osp.join(latest_folder, f'results/{model_name}-rehearsal-mmlu')
    assert osp.isdir(results_rehearsal_dir), f'{results_rehearsal_dir} not exists'
    dst_dir = f"dataset/RAIT_dataset/mmlu/{model_name}"
    os.makedirs(dst_dir, exist_ok=True)

    processor = KqKfPostProcessor(dataset_path, results_init_dir, results_rehearsal_dir, dst_dir, prompting)
    processor.process()


if __name__ == '__main__':
    main()
