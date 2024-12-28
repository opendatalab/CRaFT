If you wish to preprocess these datasets (excluding ARC) on your own, the relevant scripts are as follows.

```bash
## Download data
### TriviaQA
cd dataset/download_dataset
wget https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz
tar -zxvf triviaqa-unfiltered.tar.gz
rm triviaqa-unfiltered.tar.gz

### NQ
git clone https://github.com/google-research-datasets/natural-questions.git

### MMLU
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xvf data.tar
mv data mmlu
rm data.tar

## Preprocess data
cd ../..
PYTHONPATH=. python data_process/preprocess/proc_triviaqa.py
PYTHONPATH=. python data_process/preprocess/proc_nq.py
PYTHONPATH=. python data_process/preprocess/proc_mmlu.py
```
