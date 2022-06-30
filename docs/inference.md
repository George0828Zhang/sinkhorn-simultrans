# Inference Instructions


## Latency Evaluation (SimulEval)
1. Install a custom version of [SimulEval](./extra_installation.md). 
2. Enter the directory `eval/`. Assume that the source texts are in `test.en`, and the reference translation texts are in `test.zh`. Then run the following command for each type of models:

### offline model
```bash
bash simuleval_fullsentence.sh -m teacher_cwmt_enzh -e ../expcwmt -s ./test.en -t ./test.zh
```
### wait-k models
```bash
bash simuleval.sh \
    -a agents/simul_t2t_waitk.py \
    -m wait_1_enzh_distill \
    -k 1 \
    -e ../expcwmt \
    -s test.en \
    -t test.zh
```
### CTC-based models
```bash
bash simuleval.sh \
    -a agents/simul_t2t_ctc.py \
    -m ctc_delay3 \
    -k 3 \
    -e ../expcwmt \
    -s test.en \
    -t test.zh
```
### Results
Results should be under `eval/cwmt_zh-results`. Each subdirectory should contain the following:
```
<MODEL>.cwmt/
├── instances.log
├── prediction
└── scores.1
```

## Quality Evaluation
> **_WARNING:_**  You need to run the latency evaluation first to get the `prediction` files necessary for quality evaluation.
1. Install version 2.0 of [Sacrebleu](./extra_installation.md). 
2. Enter the directory `eval/`. 
3. Change the paths of `REF` in `run_cwmt_bleueval.sh` to your reference file paths.
4. Run
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
1. Enter the directory `eval/anticipation`
2. Assume that the source texts are in `test.en`, and the reference translation texts are in `test.zh`. 
3. Run the following
```bash
bash run_aligner.sh test en zh
```
The alignment file will appear as `./alignments/test.en-zh_1000000`. The k-AR will be printed at stdout. You can calculate k-AR with a specific k by:
```
python count_anticipation.py -k ${k} < alignments/test.en-zh_1000000
```