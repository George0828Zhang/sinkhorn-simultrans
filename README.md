# Learning to Reorder for Lower Latency Simultaneous Machine Translation
Implementation of the paper [Learning to Reorder for Lower Latency Simultaneous Machine Translation]() based on fairseq.

## Setup
1. Install fairseq
> **_WARNING:_**  Stick to the specified checkout version to avoid compatibility issues.
```bash
git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout 8b861be
python setup.py build_ext --inplace
pip install .
```
2. (Optional) [Install apex](docs/apex_installation.md) for faster mixed precision (fp16) training.
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

## Training
The output binarized files should appear under `${DATA_ROOT}/${SRC}-${TGT}/data-bin`. 

Configure environment and data path in `exp*/data_path.sh` before training, for instance:
```bash
export SRC=en
export TGT=zh
export DATA=/path/to/cwmt/en-zh/data-bin

FAIRSEQ=/path/to/fairseq
USERDIR=`realpath ../simultaneous_translation`
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"

source ~/envs/fair/bin/activate
```
Go into the directories `expcwmt/` or `expwmt15/` to start training models.
> **_NOTE:_**  We will use CWMT as example for the rest of the instructions.
## Sequence-Level KD
We need a full-sentence model as teacher for sequence-KD. 

The following command will train the teacher model.
```bash
bash 0-distill.sh
```
To distill the training set, run 
```bash
bash 0a-decode-distill.sh # generate prediction at ./mt.results/generate-test.txt
bash 0b-create-distill.sh # generate distillation data as 'train_distill_${TGT}' split.
```
To use the distillation data as training set, use/add the command line argument
```bash
--train-subset train_distill_${TGT}
```

### Distillation dataset
We provide our dataset including **distill set, pseudo reference set and reorder set** for easier reproduceability.
|CWMT En->Zh|WMT15 De->En|
|-|-|
|[Download]()|[Download]()|

## Vanilla wait-k
We can now train vanilla wait-k model as a baseline. To do this, run
```bash
DELAY=1
bash 1-vanilla_wait_k.sh ${DELAY}
```

## CTC
Train CTC baseline:
```bash
DELAY=1
bash 3-causal_ctc.sh ${DELAY}
```

## CTC + ASN
Train our proposed model:
> **_NOTE:_**  To train from scratch, remove the option `--load-pretrained-encoder-from` in `2-sinkhorn.sh`.
```bash
DELAY=1
bash 2-sinkhorn.sh ${DELAY}
```

## Pseudo Reference
The following command will generate pseudo reference from wait-9 model:
```bash
bash 5a-decode-monotonic.sh
```
The prediction will be at `monotonic.results/generate-train.txt`.
Run the following to generate pseudo reference dataset as `train_monotonic_${TGT}` split:
> **_NOTE:_** Remember to change the paths of `DATA_ROOT`,`FAIRSEQ` and `SCRIPTS` in `5b-create-monotonic.sh` to your paths.
```bash
bash 5b-create-monotonic.sh
```
To train waitk / CTC models with pseudo reference, run
```bash
DELAY=1
bash 6a-vanilla_wait_k_monotonic.sh ${DELAY}
bash 6b-causal_ctc_monotonic.sh ${DELAY}
```

## Reorder Baseline
The following command will generate word alignments and reordered target for distill set. 
> **_NOTE:_** Remember to change the path of `PREFIX` in `7a-word-align.sh` to your `cwmt/zh-en/ready/distill_${TGT}` path.
```bash
bash 7a-word-align.sh
```
The alignments will be at `./alignments.results/distill_${TGT}.${SRC}-${TGT}`.
Run the following to generate reorder dataset as `train_reorder_${TGT}` split:
> **_NOTE:_** Remember to change the paths of `DATA_ROOT`,`FAIRSEQ` in `7b-create-reorder.sh` to your paths.
```bash
bash 7b-create-reorder.sh
```
To train waitk / CTC models with reorder dataset, run
```bash
DELAY=1
bash 8a-vanilla_wait_k_reorder.sh ${DELAY}
bash 6b-causal_ctc_reorder.sh ${DELAY}
```

## Latency Evaluation (SimulEval)
Install [SimulEval](docs/extra_installation.md). Then enter the directory `eval/`.
### full-sentence model
```bash
bash simuleval_fullsentence.sh -m teacher_cwmt_enzh -e ../expcwmt -s ./test.en-zh.en -t ./test.en-zh.zh.1
```
### wait-k models
```bash
bash simuleval.sh \
    -a agents/simul_t2t_waitk.py \
    -m wait_1_enzh_distill \
    -k 1 \
    -e ../expcwmt \
    -s test.en-zh.en \
    -t test.en-zh.zh.1
```
### CTC models
```bash
bash simuleval.sh \
    -a agents/simul_t2t_ctc.py \
    -m ctc_delay3 \
    -k 3 \
    -e ../expcwmt \
    -s test.en-zh.en \
    -t test.en-zh.zh.1
```
### Results
Results should be under `eval/cwmt_zh-results`. Each subdirectory should contain the following:
```
<MODEL>.cwmt/
├── instances.log
├── prediction
└── scores.1
```
<!-- └──
├──
│    -->

## Quality Evaluation (Sacrebleu 2)
Install [Sacrebleu 2](docs/extra_installation.md). Then enter the directory `eval/`. You need to run the latency evaluation first to get the `prediction` files necessary for quality evaluation.

You can change the paths of `REF` in `run_cwmt_bleueval.sh` to your reference file paths.
```bash
bash run_cwmt_bleueval.sh
```

### Results
Results should be under `eval/cwmt_${TGT}-results/quality-results.cwmt`. Directory should contain the following:
```
quality-results.cwmt/
├── delay1-systems
├── delay3-systems
├── delay5-systems
├── delay7-systems
├── delay9-systems
└── full_sentence-systems
```


## Visualization
### Install Chinese-Simplified fonts
You need to download chinese fonts [Here](https://fonts.google.com/specimen/Noto+Sans+SC#standard-styles) first to display them correctly. Place the `NotoSansSC-Regular.otf` in the `eval/` folder.
### Install forced alignment
You need to clone and install [imputer-pytorch](https://github.com/rosinality/imputer-pytorch) to apply ctc forced alignments. 
1. Clone the repo and install
```bash
cd eval/
git clone https://github.com/rosinality/imputer-pytorch.git
cd imputer-pytorch
python setup.py install
cd ..
```
2. Copy (or link) the installation math to `eval/`
```bash
cp -r ./imputer-pytorch/torch_imputer  ./                 # copy
# ln -s ./imputer-pytorch/torch_imputer  torch_imputer    # or use symbolic link
```
### Run the Jupyter Notebook
```bash
jupyter notebook visualize_mt.ipynb
```

## k-Anticipation Rate
Suppose the test set are in the following path:
```
./
├── test.en
└── test.zh
```
To calculate k-AR, run the following
```bash
cd eval/anticipation
bash run_aligner.sh ./test en zh
```
The alignment files will be at `alignments/test.en-zh_1000000`. The k-AR will be printed at stdout. You can calculate k-AR with a specific k by:
```
python count_anticipation.py -k ${k} < alignments/test.en-zh_1000000
```