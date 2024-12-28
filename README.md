# Utilize the Flow before Stepping into the Same River Twice: Certainty Represented Knowledge Flow for Refusal-Aware Instruction Tuning

[![arXiv](https://img.shields.io/badge/arXiv-2410.06913-b31b1b.svg)](https://arxiv.org/abs/2410.06913)

[[Project Page](https://zrc007.github.io/CRaFT_page/)] [[Paper](https://arxiv.org/abs/2410.06913)]

English | [简体中文](README-zh.md)

# CRaFT: Certainty Represented Knowledge Flow for Refusal-Aware Instruction Tuning

## Introduction

Refusal-Aware Instruction Tuning (RAIT) enables Large Language Models (LLMs) to refuse to answer unknown questions. By modifying responses of unknown questions in the training data to refusal responses such as "I don't know", RAIT enhances the reliability of LLMs and reduces their hallucination. Generally, RAIT modifies training samples based on the correctness of the initial LLM's response. However, this crude approach can cause LLMs to excessively refuse answering questions they could have correctly answered, the problem we call **over-refusal**. To address this issue, we introduce Certainty Represented Knowledge Flow for Refusal-Aware Instructions Tuning (**CRaFT**). The framework of CRaFT is shown below.

<div align="center">
  <img src="./images/framework.jpg" alt="示例图片" width="600" height="300" />
</div>

## Getting Start

### Preparing enviroment and data

Our code depends on [xtuner](https://github.com/InternLM/xtuner.git) and [opencompass](https://github.com/Zrc007/opencompass_CRaFT.git), so we need to set up the corresponding environment first.

```bash
## xtuner
git clone https://github.com/InternLM/xtuner.git
cd xtuner
conda create --name xtuner-env python=3.10 -y
conda activate xtuner-env
pip install -e '.[all]'

## opencompass
git clone https://github.com/Zrc007/opencompass_CRaFT.git
cd opencompass
conda create --name opencompass python=3.10 -y
conda activate opencompass
pip install -e .
```

We conducted experiments on OEQA and MCQA separately. For OEQA, we used TriviaQA as the training set, with both TriviaQA and NQ as the test sets. For MCQA, MMLU was used as the training set, and ARC as the test set. We have already preprocessed these four datasets and stored them under path `dataset\preprocessed_dataset`.

### CRaFT

CRaFT supports both OEQA and MCQA. The process for OEQA is as follows (the execution order of MCQA is also consistent.):

- Stage1: Query Knowledge State and Flow

  ```bash
  export HF_HOME = your_HF_HOME_path
  ## get knowledge state (correctness and certainty) of initial model
  (opencompass) ./scripts/stage1/OEQA/triviaqa_kq_init.sh

  ## rehearsal train and get knowledge flow

  ### construct rehearsal training instructions
  (opencompass) ./scripts/stage1/OEQA/triviaqa_rehearsal_instructions_construction.sh

  ### rehearsal train(You can also train using other training frameworks and then add the model to `compass_config/models`)
  (xtuner-env) ./scripts/stage1/OEQA/triviaqa_rehearsal_SFT.sh
  (xtuner-env) ./scripts/stage1/OEQA/triviaqa_rehearsal_convert.sh

  ### get knowledge flow
  (opencompass) ./scripts/stage1/OEQA/triviaqa_kq_rehearsal.sh
  ```

- Stage2: Refusal-Aware instructions construction & Tuning

  ```bash
  ## Refusal-Aware instructions construction
  (opencompass) ./scripts/stage2/OEQA/triviaqa_instructions_construction.sh

  ## Refusal-Aware instructions Tuning
  (xtuner-env) ./scripts/stage2/OEQA/triviaqa_CRaFT.sh
  (xtuner-env) ./scripts/stage2/OEQA/triviaqa_CRaFT_convert.sh
  ```

## Evaluation

```bash
## OEQA
(opencompass) ./scripts/Eval/OEQA/triviaqa_eval.sh
(opencompass) ./scripts/Eval/OEQA/nq_eval.sh

## MCQA
(opencompass) ./scripts/Eval/MCQA/mmlu_eval.sh
(opencompass) ./scripts/Eval/MCQA/ARC_c_Test_eval.sh
```

## Citation

```
@article{zhu2024utilize,
  title={Utilize the Flow before Stepping into the Same River Twice: Certainty Represented Knowledge Flow for Refusal-Aware Instruction Tuning},
  author={Zhu, Runchuan and Ma, Zhipeng and Wu, Jiang and Gao, Junyuan and Wang, Jiaqi and Lin, Dahua and He, Conghui},
  journal={arXiv preprint arXiv:2410.06913},
  year={2024}
}
```
