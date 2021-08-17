export SRC=de
export TGT=en
export DATA=/livingrooms/george/wmt15/de-en/data-bin
export WANDB_START_METHOD=thread
export FAIRSEQ=~/utility/fairseq
USERDIR=`realpath ../simultaneous_translation`
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
. ~/envs/apex/bin/activate
