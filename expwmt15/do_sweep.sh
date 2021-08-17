export WANDB_START_METHOD=thread
export FAIRSEQ=~/utility/fairseq
USERDIR=`realpath ../simultaneous_translation`
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
. ~/envs/apex/bin/activate
wandb agent george0828zhang/sinkhorn-simultrans/fogr1c1z