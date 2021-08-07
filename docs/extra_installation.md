# Extra Installation
## SimulEval
We updated SimulEval with two functionalities:
1. To evaluate computational aware (CA) latency metrics for text.
2. Save actual system predictions to a file, so that multi-reference BLEU can be calculated. (we found that `instances.log` forcefully add whitespaces, which is undesired for Chinese.)
```bash
git clone https://github.com/George0828Zhang/SimulEval.git
```
Alternatively, you can use the official repository if you're skeptical of our modifications. Though you need to extract predictions manually and the result for Chinese might be inaccurate.
```bash
git clone https://github.com/facebookresearch/SimulEval.git
```
You need to add the following lines to the class `TextInstance` in `SimulEval/simuleval/scorer/instance.py` in order to obtain computational aware (CA) latency metrics:
```python
# class TextInstance(Instance):
# add following function to TextInstance
    def sentence_level_eval(self):
        super().sentence_level_eval()
        # For the computation-aware latency
        self.metrics["latency_ca"] = eval_all_latency(
            self.elapsed, self.source_length(), self.reference_length() + 1)
```
Regardless of which approach you use, proceed to install the package via pip:
```bash
cd SimulEval
pip install -e .
```

## SacreBLEU
To evaluate Translation Edit Rate (TER) or enable bootstrap resampling, we need to use SacreBLEU v2.0.0. However, version 2 currently **breaks compatibility** with the version of fairseq that we use. The solution is to use python venv to create an environment only for evaluation:
```bash
python -m venv ~/envs/sacrebleu2
```
Activate it by:
```bash
source ~/envs/sacrebleu2/bin/activate
```
Install sacrebleu version 2
```bash
git clone https://github.com/mjpost/sacrebleu.git
cd sacrebleu
pip install .
```
Then you can use sacrebleu v2, without breaking fairseq.