# Extra Installation
## SimulEval
We updated SimulEval with two functionalities:
1. To evaluate computational aware (CA) latency metrics for text.
2. Save system predictions to a file, so that multi-reference BLEU can be calculated.
```bash
git clone https://github.com/George0828Zhang/SimulEval.git
```
Alternatively, you can use the official repository if you're skeptical of our modifications.
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