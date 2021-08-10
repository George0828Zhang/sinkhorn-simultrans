# Learning to Reorder for Lower Latency Simultaneous Translation
Implementation of paper 

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

### CWMT En<->Zh
For CWMT, you need [kaggle account and api](https://www.kaggle.com/docs/api) before dowloading.
1. Setup paths in `DATA/get_data_cwmt.sh`
```bash
DATA_ROOT=/path/to/cwmt                     # set path to store raw and preprocessed data
FAIRSEQ=/path/to/fairseq                    # set path to fairseq root
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
SCRIPTS=~/utility/mosesdecoder/scripts      # set path to moses tokenizer root
source ~/envs/apex/bin/activate             # activate your virtual environment if any
```
2. Preprocess data with
```bash
cd DATA
bash get_data_cwmt.sh
```

### WMT15 De<->En
- Similarly, preprocess with `get_data_wmt15.sh`.

The output binarized files should appear under `${DATA_ROOT}/${SRC}-${TGT}/data-bin`. 

Configure environment and data path in `exp*/data_path.sh` before training, for instance:
```bash
export SRC=en
export TGT=zh
export DATA=/path/to/iwslt14/en-zh/data-bin

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
bash 0b-create-distill.sh # generate distillation data as 'train_distill_${TGT}' split from prediction
```
To use the distillation data as training set, use/add the command line argument
```bash
--train-subset train_distill_${TGT}
```

### Distillation dataset
We provide our distillation dataset for easier reproduceability.
|CWMT En->Zh|WMT15 De->En|
|-|-|
|||

## Vanilla wait-k
We can now train vanilla wait-k model as a baseline. To do this, run
> **_NOTE:_**  to train with the distillation set, set `--train-subset` to `train_distill_${TGT}` in the script.
```bash
bash 1-vanilla_wait_k.sh
```

## Sinkhorn Encoder
```bash
bash 2-sinkhorn.sh
```

## Sinkhorn wait-k
```bash
bash 4-sinkhorn-waitk.sh
```

## Online Evaluation (SimulEval)
Install [SimulEval](docs/extra_installation.md).
For wait-k models, use `simuleval_waitk.sh`.
For encoder models, use `simuleval_sinkhorn.sh`.