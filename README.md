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
export DATA=/path/to/iwslt14/en-zh/data-bin

FAIRSEQ=/path/to/fairseq
USERDIR=`realpath ../simultaneous_translation`
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"

source ~/envs/fair/bin/activate
```
Go into the directories `expcwmt/` or `expwmt15/` to start training models.

## Sequence-Level KD
We need a full-sentence model as teacher for sequence-KD. 

The following command will train the teacher model.
```bash
bash 0-distill.sh
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
We provide our dataset **including distill set** for easier reproduceability.
|CWMT En<->Zh|WMT15 De->En|
|-|-|
|[Download]()|[Download]()|

## Vanilla wait-k
We can now train vanilla wait-k model as a baseline. To do this, run
<!-- > **_NOTE:_**  to train with the distillation set, set `--train-subset` to `train_distill_${TGT}` in the script. -->
```bash
DELAY=1
bash 1-vanilla_wait_k.sh ${DELAY}
```

## CTC Encoder
```bash
DELAY=1
bash 3-causal_ctc.sh ${DELAY}
```

## CTC + ASN
```bash
DELAY=1
bash 2-sinkhorn.sh ${DELAY}
```

## Latency Evaluation (SimulEval)
Install [SimulEval](docs/extra_installation.md). Then enter the directory `eval/`.
### full-sentence
Change `EXP=../expcwmt` in `simuleval_teacher.sh` to your exp directory. Then run
```bash
bash simuleval_teacher.sh 1
```
### wait-k
Similar to above, remember to change `EXP`. Then run
```bash
DELAY=1
bash simuleval_waitk.sh waitk ${DELAY} 1
```
### CTC
Similarly, change `EXP`. Then run
```bash
DELAY=1
bash simuleval_sinkhorn.sh ctc ${DELAY} 1
```
### CTC + ASN
Similarly, change `EXP`. Then run
```bash
DELAY=1
bash simuleval_sinkhorn.sh sinkhorn ${DELAY} 1
```
### Results
Results should be under `eval/<DATA>_${TGT}-results`. Each subdirectory should contain the following:
```
<MODEL>.<DATA>/
├── instances.log
├── prediction
└── scores.1
```
<!-- └──
├──
│    -->

## Quality Evaluation (Sacrebleu 2)
Install [Sacrebleu 2](docs/extra_installation.md). Then enter the directory `eval/`. You need to run the latency evaluation first to get the `prediction` files necessary for quality evaluation.

### CWMT
You can change `SRC`, `TGT` and `DIR` in `run_cwmt_bleueval.sh` to your settings.
```bash
bash run_cwmt_bleueval.sh
```

### WMT15
Similarly change `SRC`, `TGT` and `DIR`. Then run
```bash
bash run_wmt15_bleueval.sh
```

### Results
Results should be under `eval/<DATA>_${TGT}-results/quality-results.<DATA>`. Directory should contain the following:
```
quality-results.<DATA>/
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
You need to clone and install [imputer-pytorch](https://github.com/rosinality/imputer-pytorch) to apply ctc force alignments. 
1. Clone the repo and install
```bash
git clone https://github.com/rosinality/imputer-pytorch.git
cd imputer-pytorch
python setup.py install
```
2. Add installation path to system path in `visualize_mt.ipynb`.
```bash
sys.path.insert(0, "~/utility/imputer-pytorch")
```
### Run the Jupyter Notebook
```bash
jupyter notebook visualize_mt.ipynb
```