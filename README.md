# Simultaneous Translation with Gumbel Sinkhorn Attention 
Proposed: Learning to translate monotonically by optimal transport.

## Setup

1. Install fairseq
```bash
git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout 8b861be
python setup.py build_ext --inplace
```
2. (Optional) [Install](docs/apex_installation.md) apex for faster mixed precision (fp16) training.
3. Install dependencies
```bash
pip install -r requirements.txt
```

## Data Preparation
This section introduces the data preparation for training and evaluation.

First download moses tokenizer:
```bash
git clone https://github.com/moses-smt/mosesdecoder.git
```

### iwslt14 de<->en

1. Setup paths in `DATA/get_data_iwslt14_deen.sh`
```bash
DATA_ROOT=/path/to/iwslt14          # set path to store raw and preprocessed data
FAIRSEQ=/path/to/fairseq                  # set path to fairseq root
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
SCRIPTS=~/utility/mosesdecoder/scripts    # set path to moses tokenizer root
source ~/envs/apex/bin/activate           # activate your virtual environment if any
```
2. Preprocess data with
```bash
cd DATA
bash get_data_iwslt14_deen.sh
```

### wmt15 de<->en
- Similarly, preprocess with `get_data_wmt15.sh`.


The output binarized files should appear under `${DATA_ROOT}/de-en/data-bin`. 

Configure environment and data path in `exp/data_path.sh` before training:
```bash
export SRC=de
export TGT=en
export DATA=/path/to/iwslt14/de-en/data-bin

FAIRSEQ=/path/to/fairseq
USERDIR=`realpath ../simultaneous_translation`
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"

source ~/envs/fair/bin/activate
```

## Sequence-Level KD
We need a machine translation model as teacher for sequence-KD. 

The following command will train the teacher model.
```bash
cd exp
bash 0-distill.sh
```
Validate the performance with
```bash
cd eval
bash eval_mt.sh
```
To distill the training set, run 
```bash
bash 0a-decode-distill.sh # generate prediction at ./mt.results/generate-test.txt
bash 0b-create-distill-tsv.sh # generate distillation data as 'train_distill' split from prediction
```
To use the distillation data as training set, use/add the command line argument
```bash
--train-subset train_distill
```

### Pretrained models & distillation dataset
|iwslt14 de-en|wmt15 de-en|
|-|-|
|[model (31.62)](https://onedrive.live.com/download?cid=3E549F3B24B238B4&resid=3E549F3B24B238B4%216345&authkey=ANTgZvmncA2OFt0)|model|
|[distilled (de->en)](https://onedrive.live.com/download?cid=3E549F3B24B238B4&resid=3E549F3B24B238B4%216343&authkey=ADWEb0KVv4MwMqo)|distilled (de->en)|

## Vanilla wait-k
We can now train vanilla wait-k ST model as a baseline. To do this, run
> **_NOTE:_**  to train with the distillation set, set `--train-subset` to `distill_st` in the script.
```bash
bash 2-vanilla_wait_k.sh
```
### Pretrained models
The gcmvn and spm share the same files with corresponding poretrained asr.
|DATA|arch|en-es|en-de|
|-|-|-|-|
||wait-1||||
||wait-3||||
||wait-5||||
||wait-7||||
||wait-9||||


## Offline Evaluation (BLEU only)
## Online Evaluation (SimulEval)
Install [SimulEval](docs/extra_installation.md).