# Extra Installation
## SimulEval
```bash
git clone https://github.com/facebookresearch/SimulEval.git
```
To evaluate computational aware (CA) latency metrics, you need to add the following lines to the class `TextInstance` in `SimulEval/simuleval/scorer/instance.py`:
```python
# class TextInstance(Instance):
# add following function to TextInstance
    def sentence_level_eval(self):
        super().sentence_level_eval()
        # For the computation-aware latency
        self.metrics["latency_ca"] = eval_all_latency(
            self.elapsed, self.source_length(), self.reference_length() + 1)
```
Proceed to install the package via pip:
```bash
cd SimulEval
pip install -e .
```