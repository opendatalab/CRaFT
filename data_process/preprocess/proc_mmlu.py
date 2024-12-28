import os
import os.path as osp
import pandas as pd
import json
from mmengine import load

# 定义转换后的列名顺序
converted_cols = ['id', 'lang', 'question', 'A', 'B', 'C', 'D', 'target']

def convert(src_root_dir, dst_root_dir):
    """
    将源目录中的数据文件转换为目标格式，并保存到目标目录中。

    参数:
    src_root_dir (str): 源数据的根目录
    dst_root_dir (str): 目标数据的根目录
    """
    
    # 遍历不同的数据集划分
    for split in ['dev', 'val', 'test']:
        # 源目录和目标目录的路径
        src_dir = osp.join(src_root_dir, split)
        dst_dir = osp.join(dst_root_dir, split)

        # 确保源目录存在
        assert osp.isdir(src_dir)
        # 如果目标目录不存在，则创建它
        os.makedirs(dst_dir, exist_ok=True)

        # 获取任务名称，去除扩展名和split后缀
        task_names = [
            osp.splitext(_)[0] for _ in os.listdir(src_dir)
            if _.endswith('.csv')
        ]
        # 确保每个任务名都以split后缀结尾
        assert all([_.endswith('_' + split) for _ in task_names])
        # 去除split后缀以获取纯任务名
        task_names = [_[:-len('_' + split)] for _ in task_names]

        # 遍历每个任务
        for task_name in task_names:
            # 源文件路径和目标文件路径
            src_file = osp.join(src_dir, f'{task_name}_{split}.csv')
            dst_file = osp.join(dst_dir, f'{task_name}_{split}.csv')

            # 读取CSV文件为DataFrame
            data_df = pd.read_csv(src_file, header=None)

            # 用'None'填充空缺值
            data_df.fillna('None', inplace=True)

            # 生成唯一标识符
            ids = [f'{task_name}/{split}/{i}' for i in range(len(data_df))]

            # 设置列名并添加ID和语言信息
            data_df.columns = ['question', 'A', 'B', 'C', 'D', 'target']
            data_df['id'] = ids
            data_df['lang'] = 'en'
            # 调整列的顺序
            data_df = data_df[converted_cols]

            # 将转换后的DataFrame保存为CSV文件
            data_df.to_csv(dst_file, index=False)

if __name__ == '__main__':
    # 源目录和目标目录的路径
    src_dir = 'dataset/download_dataset/mmlu'
    dst_dir = 'dataset/preprocessed_dataset/mmlu'

    # 删除目标目录及其内容（如果存在）
    os.system(f"rm -rf {dst_dir}")
    # 执行转换函数
    convert(src_dir, dst_dir)
