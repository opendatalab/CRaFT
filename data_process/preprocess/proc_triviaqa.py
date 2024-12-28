import os
import os.path as osp
import pandas as pd
from mmengine import load
from utils import df_to_list_of_dict, dump_to_json_file

def proc_triviaqa():
    # 定义源数据和目标数据的目录
    src_dir = 'dataset/download_dataset/triviaqa-unfiltered'
    dst_dir = 'dataset/preprocessed_dataset/triviaqa/en'
    
    # 如果目标目录不存在，则创建
    os.makedirs(dst_dir, exist_ok=True)

    # 遍历数据集的两个分割：dev 和 train
    for split in ['dev', 'train']:
        # 定义源文件路径和目标文件路径（CSV和JSON格式）
        src_file = osp.join(src_dir, f'unfiltered-web-{split}.json')
        dst_csv = osp.join(dst_dir, f'{split}.csv')
        dst_json = osp.join(dst_dir, f'{split}.json')

        # 读取源JSON文件并转换为DataFrame
        df = pd.DataFrame(load(src_file)['Data'])

        # 重命名列名以便更容易理解
        df.rename(
            columns={
                'Question': 'question',
                'QuestionId': 'qid',
            },
            inplace=True,
        )
        
        # 提取答案的主要值，并创建包含所有可能答案的列表
        df['answer'] = df['Answer'].apply(lambda x: x['Value'])
        df['answers'] = df['Answer'].apply(lambda x: list(
            set([x['Value'], x['NormalizedValue']] + x['Aliases'] + x['NormalizedAliases'])))
        
        # 添加语言信息列
        df['lang'] = 'en'
        
        # 选择需要的列
        df = df[['qid', 'lang', 'question', 'answer', 'answers']]

        # 将DataFrame保存为CSV文件
        df.to_csv(dst_csv, index=False)
        
        # 将DataFrame转换为列表字典并保存为JSON文件
        dump_to_json_file(df_to_list_of_dict(df), dst_json)

# 主函数调用，用于处理triviaqa数据集
if __name__ == '__main__':
    proc_triviaqa()
